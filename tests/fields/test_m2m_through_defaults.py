"""
Tests for ManyToManyRelation.add(..., through_defaults=...).
"""

import datetime
import os

import pytest
import pytest_asyncio

from tests.fields import through_models as tm
from tortoise.context import tortoise_test_context
from tortoise.exceptions import ConfigurationError, OperationalError, ValidationError


@pytest_asyncio.fixture(scope="function")
async def db_through():
    db_url = os.getenv("TORTOISE_TEST_DB", "sqlite://:memory:")
    async with tortoise_test_context(
        modules=["tests.fields.through_models"],
        db_url=db_url,
        app_label="models",
        connection_label="models",
    ) as ctx:
        yield ctx


async def make_club_with_members(count=2):
    club = await tm.Club.create(name="club")
    members = [await tm.Member.create(name=f"member{i}") for i in range(count)]
    return club, members


@pytest.mark.asyncio
async def test_add_with_through_defaults(db_through):
    club, (member,) = await make_club_with_members(1)
    joined = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    await club.members.add(member, through_defaults={"role": "admin", "joined_at": joined})

    assert await club.members.all() == [member]
    membership = await tm.Membership.get()
    assert membership.role == "admin"
    assert membership.joined_at == joined
    assert membership.club_id == club.id
    assert membership.member_id == member.id


@pytest.mark.asyncio
async def test_add_broadcasts_single_dict(db_through):
    club, members = await make_club_with_members(2)
    await club.members.add(*members, through_defaults={"role": "member"})

    roles = await tm.Membership.all().order_by("member_id").values_list("role", flat=True)
    assert roles == ["member", "member"]


@pytest.mark.asyncio
async def test_add_per_instance_defaults(db_through):
    club, members = await make_club_with_members(3)
    await club.members.add(
        *members,
        through_defaults=[{"role": "admin"}, {"role": "member"}, {"role": None}],
    )

    rows = await tm.Membership.all().order_by("member_id").values_list("member_id", "role")
    assert rows == [
        (members[0].id, "admin"),
        (members[1].id, "member"),
        (members[2].id, None),
    ]


@pytest.mark.asyncio
async def test_add_per_instance_none_entries(db_through):
    club, members = await make_club_with_members(2)
    await club.members.add(*members, through_defaults=[{"role": "admin"}, None])

    rows = await tm.Membership.all().order_by("member_id").values_list("role", flat=True)
    assert rows == ["admin", None]


@pytest.mark.asyncio
async def test_add_duplicate_does_not_update_through_values(db_through):
    club, (member,) = await make_club_with_members(1)
    await club.members.add(member, through_defaults={"role": "admin"})
    # Re-adding is idempotent and must not silently update the existing row.
    await club.members.add(member, through_defaults={"role": "member"})

    assert await tm.Membership.all().count() == 1
    membership = await tm.Membership.get()
    assert membership.role == "admin"


@pytest.mark.asyncio
async def test_add_batch_with_duplicate_only_inserts_missing(db_through):
    club, members = await make_club_with_members(2)
    await club.members.add(members[0], through_defaults={"role": "admin"})
    await club.members.add(*members, through_defaults={"role": "member"})

    rows = await tm.Membership.all().order_by("member_id").values_list("role", flat=True)
    assert rows == ["admin", "member"]


@pytest.mark.asyncio
async def test_callable_defaults_evaluated_once_per_insert(db_through):
    club, members = await make_club_with_members(2)
    calls = []

    def make_joined():
        calls.append(1)
        return datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc)

    await club.members.add(*members, through_defaults={"joined_at": make_joined})
    assert len(calls) == 2

    # Existing relations are skipped without evaluating callables again.
    await club.members.add(*members, through_defaults={"joined_at": make_joined})
    assert len(calls) == 2

    member3 = await tm.Member.create(name="member3")
    await club.members.add(member3, through_defaults={"joined_at": make_joined})
    assert len(calls) == 3

    joined_values = await tm.Membership.all().values_list("joined_at", flat=True)
    assert all(value is not None for value in joined_values)


@pytest.mark.asyncio
async def test_invalid_value_aborts_whole_batch(db_through):
    club, members = await make_club_with_members(2)
    with pytest.raises(ValidationError):
        await club.members.add(
            *members,
            through_defaults=[{"role": "ok", "level": 1}, {"role": "bad", "level": -1}],
        )

    # No relation and no through rows may be left behind.
    assert await club.members.all() == []
    assert await tm.Membership.all().count() == 0


@pytest.mark.asyncio
async def test_unknown_through_field_raises(db_through):
    club, (member,) = await make_club_with_members(1)
    with pytest.raises(ConfigurationError):
        await club.members.add(member, through_defaults={"no_such_field": 1})
    assert await tm.Membership.all().count() == 0


@pytest.mark.asyncio
async def test_relation_keys_cannot_be_set_via_through_defaults(db_through):
    club, (member,) = await make_club_with_members(1)
    for key in ("club_id", "member_id", "club"):
        with pytest.raises(ConfigurationError):
            await club.members.add(member, through_defaults={key: 1})
    assert await tm.Membership.all().count() == 0


@pytest.mark.asyncio
async def test_through_defaults_length_must_match_instances(db_through):
    club, members = await make_club_with_members(2)
    with pytest.raises(ConfigurationError):
        await club.members.add(*members, through_defaults=[{"role": "admin"}])
    assert await tm.Membership.all().count() == 0


@pytest.mark.asyncio
async def test_per_instance_dicts_must_have_same_keys(db_through):
    club, members = await make_club_with_members(2)
    with pytest.raises(ConfigurationError):
        await club.members.add(*members, through_defaults=[{"role": "admin"}, {"level": 1}])
    assert await tm.Membership.all().count() == 0


@pytest.mark.asyncio
async def test_through_defaults_requires_through_model(db_through):
    one = await tm.PlainA.create()
    two = await tm.PlainB.create()
    with pytest.raises(ConfigurationError):
        await one.bees.add(two, through_defaults={"anything": 1})


@pytest.mark.asyncio
async def test_add_unsaved_instance_still_raises(db_through):
    club, _ = await make_club_with_members(0)
    with pytest.raises(OperationalError):
        await club.members.add(tm.Member(name="x"), through_defaults={"role": "admin"})


@pytest.mark.asyncio
async def test_reverse_direction_add_with_through_defaults(db_through):
    club, (member,) = await make_club_with_members(1)
    await member.clubs.add(club, through_defaults={"role": "owner"})

    membership = await tm.Membership.get()
    assert membership.role == "owner"
    # FK direction must not be swapped.
    assert membership.club_id == club.id
    assert membership.member_id == member.id
    assert await member.clubs.all() == [club]


@pytest.mark.asyncio
async def test_self_referencing_add_with_through_defaults(db_through):
    node1 = await tm.Node.create(name="n1")
    node2 = await tm.Node.create(name="n2")
    await node1.links.add(node2, through_defaults={"label": "next"})

    link = await tm.NodeLink.get()
    assert link.label == "next"
    assert link.from_node_id == node1.id
    assert link.to_node_id == node2.id

    # Reverse direction writes the swapped columns.
    node3 = await tm.Node.create(name="n3")
    await node2.linked_from.add(node3, through_defaults={"label": "prev"})
    link2 = await tm.NodeLink.get(from_node_id=node3.id)
    assert link2.label == "prev"
    assert link2.to_node_id == node2.id


@pytest.mark.asyncio
async def test_using_db_is_respected(db_through):
    club, (member,) = await make_club_with_members(1)
    db = db_through.db()
    await club.members.add(member, using_db=db, through_defaults={"role": "admin"})
    assert (await tm.Membership.get()).role == "admin"


@pytest.mark.asyncio
async def test_plain_add_still_works_with_through_model(db_through):
    club, (member,) = await make_club_with_members(1)
    await club.members.add(member)

    assert await club.members.all() == [member]
    membership = await tm.Membership.get()
    assert membership.role is None
    assert membership.joined_at is None


@pytest.mark.asyncio
async def test_fetch_remove_and_clear_still_work(db_through):
    club, members = await make_club_with_members(2)
    await club.members.add(*members, through_defaults=[{"role": "admin"}, {"role": "member"}])

    await club.fetch_related("members")
    assert list(club.members) == members

    await club.members.remove(members[0])
    assert await tm.Membership.all().count() == 1
    assert (await tm.Membership.get()).role == "member"

    await club.members.clear()
    assert await tm.Membership.all().count() == 0
    assert await club.members.all() == []


@pytest.mark.asyncio
async def test_plain_m2m_without_extra_fields_unaffected(db_through):
    one = await tm.PlainA.create()
    two1 = await tm.PlainB.create()
    two2 = await tm.PlainB.create()
    await one.bees.add(two1, two2)
    await one.bees.add(two1)  # idempotent
    assert await one.bees.all() == [two1, two2]
    await one.bees.remove(two1)
    assert await one.bees.all() == [two2]
