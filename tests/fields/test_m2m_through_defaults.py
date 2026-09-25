"""
Tests for ManyToManyRelation.add() with through_defaults on explicit through models.
"""

import datetime

import pytest
import pytest_asyncio

from tests.fields.through_defaults_models import (
    Club,
    ClubMembership,
    Member,
    Node,
    NodeLink,
    Tag,
)
from tortoise.context import tortoise_test_context
from tortoise.exceptions import FieldError, OperationalError, ParamsError, ValidationError


@pytest_asyncio.fixture(scope="function")
async def db_through():
    async with tortoise_test_context(
        modules=["tests.fields.through_defaults_models"],
        app_label="models",
        connection_label="models",
    ) as ctx:
        yield ctx


async def make_club_and_members(count=2):
    club = await Club.create(name="Chess")
    members = [await Member.create(name=f"m{i}") for i in range(count)]
    return club, members


@pytest.mark.asyncio
async def test_add_with_through_defaults(db_through):
    club, (member,) = await make_club_and_members(1)
    await club.members.add(member, through_defaults={"role": "admin", "priority": 3})
    assert await club.members.all() == [member]
    membership = await ClubMembership.get(club=club, member=member)
    assert membership.role == "admin"
    assert membership.priority == 3
    assert membership.joined_at is None


@pytest.mark.asyncio
async def test_add_batch_per_object_defaults(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    await club.members.add(
        (m1, {"role": "admin", "priority": 10}),
        (m2, {"role": "member"}),
        through_defaults={"priority": 1},
    )
    assert await club.members.all().count() == 2
    ms1 = await ClubMembership.get(club=club, member=m1)
    ms2 = await ClubMembership.get(club=club, member=m2)
    # per-object value wins over shared through_defaults
    assert (ms1.role, ms1.priority) == ("admin", 10)
    # shared through_defaults applies where no per-object value is given
    assert (ms2.role, ms2.priority) == ("member", 1)


@pytest.mark.asyncio
async def test_add_mixed_plain_and_tuple_instances(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    await club.members.add((m1, {"role": "admin"}), m2, through_defaults={"role": "member"})
    assert (await ClubMembership.get(club=club, member=m1)).role == "admin"
    assert (await ClubMembership.get(club=club, member=m2)).role == "member"


@pytest.mark.asyncio
async def test_add_idempotent_does_not_update_existing(db_through):
    club, (member,) = await make_club_and_members(1)
    await club.members.add(member, through_defaults={"role": "admin"})
    # adding again must be silently ignored and must NOT update the stored role
    await club.members.add(member, through_defaults={"role": "moderator"})
    assert await club.members.all().count() == 1
    assert (await ClubMembership.get(club=club, member=member)).role == "admin"


@pytest.mark.asyncio
async def test_add_partial_batch_only_inserts_missing(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    await club.members.add(m1, through_defaults={"role": "admin"})
    await club.members.add(m1, (m2, {"role": "member"}), through_defaults={"role": "changed"})
    assert await club.members.all().count() == 2
    # pre-existing relation keeps its original through values
    assert (await ClubMembership.get(club=club, member=m1)).role == "admin"
    assert (await ClubMembership.get(club=club, member=m2)).role == "member"


@pytest.mark.asyncio
async def test_callable_defaults_evaluated_once_per_insert(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    calls = []

    def make_role():
        calls.append(1)
        return "admin"

    await club.members.add(m1, m2, through_defaults={"role": make_role})
    assert len(calls) == 2
    # already existing relations: no insert happens, callables are not evaluated
    await club.members.add(m1, m2, through_defaults={"role": make_role})
    assert len(calls) == 2
    # only the missing relation is inserted, so exactly one more evaluation
    m3 = await Member.create(name="m3")
    await club.members.add(m1, m3, through_defaults={"role": make_role})
    assert len(calls) == 3
    assert await club.members.all().count() == 3


@pytest.mark.asyncio
async def test_callable_default_value_is_validated(db_through):
    club, (m1, m2) = await make_club_and_members(2)

    def bad_priority():
        return -1

    with pytest.raises(ValidationError):
        await club.members.add(
            m1, (m2, {"priority": bad_priority}), through_defaults={"role": "member"}
        )
    assert await club.members.all().count() == 0
    assert await ClubMembership.all().count() == 0


@pytest.mark.asyncio
async def test_invalid_value_in_batch_leaves_no_partial_relations(db_through):
    club, (m1, m2, m3) = await make_club_and_members(3)
    with pytest.raises(ValidationError):
        await club.members.add(
            (m1, {"role": "admin"}),
            (m2, {"role": "member", "priority": -5}),
            (m3, {"role": "member"}),
        )
    # validation of the whole batch happens before any insert
    assert await club.members.all().count() == 0
    assert await ClubMembership.all().count() == 0


@pytest.mark.asyncio
async def test_unknown_through_field_raises_and_writes_nothing(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    with pytest.raises(FieldError):
        await club.members.add(
            (m1, {"role": "admin"}), (m2, {"role": "member", "nope": 1})
        )
    assert await club.members.all().count() == 0
    assert await ClubMembership.all().count() == 0


@pytest.mark.asyncio
async def test_through_defaults_cannot_override_relation_keys(db_through):
    club, (member,) = await make_club_and_members(1)
    with pytest.raises(FieldError):
        await club.members.add(member, through_defaults={"role": "x", "member_id": 999})
    assert await ClubMembership.all().count() == 0


@pytest.mark.asyncio
async def test_through_defaults_without_through_model_raises(db_through):
    club = await Club.create(name="Chess")
    tag = await Tag.create(name="fun")
    with pytest.raises(OperationalError, match="explicit through model"):
        await club.tags.add(tag, through_defaults={"role": "admin"})
    # plain add on the auto-generated through table still works
    await club.tags.add(tag)
    assert await club.tags.all() == [tag]


@pytest.mark.asyncio
async def test_add_from_reverse_direction(db_through):
    club, (member,) = await make_club_and_members(1)
    await member.clubs.add(club, through_defaults={"role": "admin", "priority": 2})
    assert await member.clubs.all() == [club]
    membership = await ClubMembership.get(club=club, member=member)
    assert (membership.role, membership.priority) == ("admin", 2)


@pytest.mark.asyncio
async def test_self_referencing_m2m(db_through):
    node1 = await Node.create(name="n1")
    node2 = await Node.create(name="n2")
    node3 = await Node.create(name="n3")
    await node1.links.add((node2, {"weight": 2.5}), node3, through_defaults={"weight": 0.5})
    assert await node1.links.all().count() == 2
    assert (await NodeLink.get(from_node=node1, to_node=node2)).weight == 2.5
    assert (await NodeLink.get(from_node=node1, to_node=node3)).weight == 0.5
    # reverse direction sees the same rows
    assert await node2.linked_from.all() == [node1]


@pytest.mark.asyncio
async def test_using_db_is_respected(db_through):
    club, (member,) = await make_club_and_members(1)
    db = db_through.db()
    await club.members.add(member, through_defaults={"role": "admin"}, using_db=db)
    membership = await ClubMembership.filter(club=club, member=member).using_db(db).get()
    assert membership.role == "admin"


@pytest.mark.asyncio
async def test_remove_and_clear_still_work(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    await club.members.add(
        (m1, {"role": "admin"}), (m2, {"role": "member"})
    )
    await club.members.remove(m1)
    assert await club.members.all() == [m2]
    await club.members.clear()
    assert await club.members.all().count() == 0
    assert await ClubMembership.all().count() == 0


@pytest.mark.asyncio
async def test_fetch_related_and_filtering_still_work(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    await club.members.add((m1, {"role": "admin"}), (m2, {"role": "member"}))
    fetched = await Club.get(name="Chess").prefetch_related("members")
    assert set(fetched.members) == {m1, m2}
    assert await club.members.filter(name="m0") == [m1]


@pytest.mark.asyncio
async def test_datetime_and_default_fallback_for_missing_columns(db_through):
    club, (m1, m2) = await make_club_and_members(2)
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    await club.members.add(
        (m1, {"role": "admin", "joined_at": now}),
        (m2, {"role": "member"}),
    )
    ms1 = await ClubMembership.get(club=club, member=m1)
    ms2 = await ClubMembership.get(club=club, member=m2)
    assert ms1.joined_at == now
    # joined_at not given for m2: nullable field falls back to NULL
    assert ms2.joined_at is None
    # priority not given for either: field default is applied
    assert (ms1.priority, ms2.priority) == (0, 0)


@pytest.mark.asyncio
async def test_malformed_arguments(db_through):
    club, (member,) = await make_club_and_members(1)
    with pytest.raises(ParamsError):
        await club.members.add(member, through_defaults="role")
    with pytest.raises(ParamsError):
        await club.members.add((member, "role"))
    assert await ClubMembership.all().count() == 0
