import asyncio

import pytest

from tests.testmodels import Event, Team, Tournament
from tortoise import identity_scope
from tortoise.transactions import in_transaction


@pytest.mark.asyncio
async def test_identity_scope_reuses_instances_and_preserves_local_changes(db_isolated):
    tournament = await Tournament.create(name="initial", desc="old description")

    async with identity_scope():
        full = await Tournament.get(pk=tournament.pk)
        full.name = "local name"

        partial = await Tournament.filter(pk=tournament.pk).only("desc").first()
        assert partial is full
        assert partial.desc == "old description"
        assert partial.name == "local name"

        supplemented = await Tournament.filter(pk=tournament.pk).only("name").first()
        assert supplemented is full
        assert supplemented.desc == "old description"
        assert supplemented.name == "local name"

        await Tournament.filter(pk=tournament.pk).update(
            name="database name", desc="new description"
        )
        refetched = await Tournament.get(pk=tournament.pk)
        assert refetched is full
        assert refetched.name == "local name"
        assert refetched.desc == "new description"


@pytest.mark.asyncio
async def test_without_identity_scope_queries_construct_separate_instances(db_isolated):
    tournament = await Tournament.create(name="initial")

    first = await Tournament.get(pk=tournament.pk)
    second = await Tournament.get(pk=tournament.pk)

    assert first is not second


@pytest.mark.asyncio
async def test_identity_scope_select_related_reuses_instance(db_isolated):
    async with identity_scope():
        tournament = await Tournament.create(name="tournament")
        other_tournament = await Tournament.create(name="other tournament")
        event = await Event.create(name="event", tournament=tournament)

        selected = await Event.get(pk=event.pk).select_related("tournament")
        selected.tournament = other_tournament
        only_selected = await (
            Event.filter(pk=event.pk)
            .only("event_id", "name", "tournament__id", "tournament__name")
            .select_related("tournament")
            .first()
        )

        assert only_selected is selected
        assert only_selected.tournament is other_tournament
        assert only_selected.tournament_id == other_tournament.pk


@pytest.mark.asyncio
async def test_identity_scope_prefetch_related_reuses_instance(db_isolated):
    async with identity_scope():
        tournament = await Tournament.create(name="tournament")
        event = await Event.create(name="event", tournament=tournament)
        team = await Team.create(name="team")
        await event.participants.add(team)

        prefetched_event = await Event.get(pk=event.pk).prefetch_related(
            "tournament", "participants"
        )
        prefetched_tournament = await Tournament.get(pk=tournament.pk).prefetch_related(
            "events__participants"
        )
        prefetched_team = await Team.get(pk=team.pk)

        assert prefetched_event.tournament is tournament
        assert prefetched_event.tournament is prefetched_tournament
        assert list(prefetched_event.participants) == [prefetched_team]
        assert list(prefetched_tournament.events) == [prefetched_event]


@pytest.mark.asyncio
async def test_concurrent_identity_scopes_have_independent_instances(db_isolated):
    tournament = await Tournament.create(name="initial")

    async def read():
        async with identity_scope():
            instance = await Tournament.get(pk=tournament.pk)
            await asyncio.sleep(0.02)
            again = await Tournament.get(pk=tournament.pk)
            assert again is instance
            return instance

    first, second = await asyncio.gather(read(), read())
    assert first is not second


@pytest.mark.asyncio
async def test_nested_identity_scope_reuses_parent_instance(db_isolated):
    tournament = await Tournament.create(name="initial")

    async with identity_scope():
        outer = await Tournament.get(pk=tournament.pk)

        async with identity_scope():
            inner = await Tournament.get(pk=tournament.pk)
            assert inner is outer


@pytest.mark.asyncio
async def test_identity_scope_delete_evicts_instance(db_isolated):
    tournament = await Tournament.create(name="initial")

    async with identity_scope():
        cached = await Tournament.get(pk=tournament.pk)
        await cached.delete()
        assert await Tournament.get_or_none(pk=tournament.pk) is None

        replacement = await Tournament.create(name="replacement")
        await Tournament.filter(pk=replacement.pk).delete()
        assert await Tournament.get_or_none(pk=replacement.pk) is None


@pytest.mark.asyncio
async def test_identity_scope_refresh_bypasses_cache(db_isolated):
    tournament = await Tournament.create(name="initial", desc="initial description")

    async with identity_scope():
        cached = await Tournament.get(pk=tournament.pk)
        cached.name = "local name"
        await Tournament.filter(pk=tournament.pk).update(name="database name")

        await cached.refresh_from_db(fields=["name"])

        assert cached.name == "database name"
        assert cached.desc == "initial description"


@pytest.mark.asyncio
async def test_identity_scope_rollback_evicts_cached_instances(db_isolated):
    tournament = await Tournament.create(name="initial")

    async with identity_scope():
        cached = await Tournament.get(pk=tournament.pk)
        try:
            async with in_transaction():
                await Tournament.filter(pk=tournament.pk).update(name="transaction name")
                changed = await Tournament.get(pk=tournament.pk)
                assert changed is cached
                assert cached.name == "transaction name"
                raise RuntimeError("rollback")
        except RuntimeError:
            pass

        after_rollback = await Tournament.get(pk=tournament.pk)
        assert after_rollback is not cached
        assert after_rollback.name == "initial"
