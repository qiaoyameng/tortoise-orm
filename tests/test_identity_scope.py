import asyncio

import pytest
import pytest_asyncio

from tortoise import fields
from tortoise.context import tortoise_test_context
from tortoise.exceptions import DoesNotExist
from tortoise.identity import IdentityScope, current_scope, identity_scope
from tortoise.models import Model


class Tournament(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)
    desc = fields.TextField(null=True)

    events: fields.ReverseRelation["Event"]

    class Meta:
        table = "identity_tournament"


class Event(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)
    tournament: fields.ForeignKeyRelation[Tournament] = fields.ForeignKeyField(
        "models.Tournament", related_name="events"
    )
    participants: fields.ManyToManyRelation["Team"] = fields.ManyToManyField(
        "models.Team", related_name="events"
    )

    class Meta:
        table = "identity_event"


class Team(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)

    events: fields.ManyToManyRelation[Event]

    class Meta:
        table = "identity_team"


@pytest_asyncio.fixture(scope="module")
async def db_module():
    """Module-scoped context with a small, dedicated model set (fast schema)."""
    async with tortoise_test_context(
        modules=["tests.test_identity_scope"],
        app_label="models",
        connection_label="models",
    ) as ctx:
        yield ctx


@pytest_asyncio.fixture(autouse=True, scope="session")
async def _event_loop_heartbeat():
    # Some sandboxed environments do not deliver cross-thread event-loop
    # wakeups (loop.call_soon_threadsafe, used by aiosqlite) while the loop is
    # parked in select(). A lightweight heartbeat keeps the loop pumping so
    # aiosqlite completions are processed. It is a no-op elsewhere.
    async def beat() -> None:
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(beat())
    yield
    task.cancel()


@pytest.mark.asyncio
async def test_no_scope_default_behaviour(db):
    tournament = await Tournament.create(name="T1")
    a = await Tournament.get(id=tournament.id)
    b = await Tournament.get(id=tournament.id)
    assert a == b
    assert a is not b


@pytest.mark.asyncio
async def test_same_instance_within_scope(db):
    tournament = await Tournament.create(name="T1")
    with identity_scope():
        a = await Tournament.get(id=tournament.id)
        b = await Tournament.get(id=tournament.id)
        c = await Tournament.filter(name="T1").first()
        assert a is b
        assert a is c
    # After the scope exits, fresh instances are constructed again.
    d = await Tournament.get(id=tournament.id)
    assert d is not a


@pytest.mark.asyncio
async def test_interleaved_full_only_and_related_queries(db):
    tournament = await Tournament.create(name="T1", desc="original")
    await Event.create(name="E1", tournament=tournament)
    team = await Team.create(name="Team1")
    event = await Event.get(name="E1")
    await event.participants.add(team)

    with identity_scope():
        # 1. Partial query first: only pk + "name" are loaded.
        partial = await Tournament.get(id=tournament.id).only("id", "name")
        assert partial._partial
        assert partial.name == "T1"
        with pytest.raises(AttributeError):
            _ = partial.desc

        # 2. Local modification must survive later queries.
        partial.name = "local edit"

        # 3. Full query returns the same instance, fills missing fields,
        #    but does not overwrite the local modification.
        full = await Tournament.get(id=tournament.id)
        assert full is partial
        assert full.name == "local edit"
        assert full.desc == "original"
        assert not full._partial

        # 4. select_related participates in the same identity map.
        event = await Event.get(name="E1").select_related("tournament")
        assert event.tournament is full

        # 5. prefetch_related (reverse FK) participates as well.
        fetched = await Tournament.get(id=tournament.id).prefetch_related("events")
        assert fetched is full
        assert fetched.events[0] is event

        # 6. m2m prefetch participates too.
        direct_team = await Team.get(id=team.id)
        event2 = await Event.get(name="E1").prefetch_related("participants")
        assert event2 is event
        assert event2.participants[0] is direct_team


@pytest.mark.asyncio
async def test_unmodified_fields_are_refreshed(db):
    tournament = await Tournament.create(name="T1", desc="d1")
    with identity_scope():
        obj = await Tournament.get(id=tournament.id)
        obj.name = "local edit"
        # Another writer changes the row underneath us.
        await Tournament.filter(id=tournament.id).update(name="db name", desc="db desc")
        again = await Tournament.get(id=tournament.id)
        assert again is obj
        # Locally modified field is preserved...
        assert obj.name == "local edit"
        # ...but untouched fields pick up the newer database values.
        assert obj.desc == "db desc"


@pytest.mark.asyncio
async def test_save_resets_modification_baseline(db):
    tournament = await Tournament.create(name="T1")
    with identity_scope():
        obj = await Tournament.get(id=tournament.id)
        obj.name = "saved name"
        await obj.save()
        # After save(), the saved value is the new baseline: a later query may
        # refresh it with a newer database value.
        await Tournament.filter(id=tournament.id).update(name="other")
        again = await Tournament.get(id=tournament.id)
        assert again is obj
        assert obj.name == "other"


@pytest.mark.asyncio
async def test_create_inside_scope_is_canonical(db):
    with identity_scope():
        created = await Tournament.create(name="T1")
        fetched = await Tournament.get(id=created.id)
        assert fetched is created


@pytest.mark.asyncio
async def test_delete_evicts_from_scope(db):
    await Tournament.create(id=100, name="T1")
    with identity_scope():
        first = await Tournament.get(id=100)
        await first.delete()
        with pytest.raises(DoesNotExist):
            await Tournament.get(id=100)
        # A new row with the same pk must produce a new instance, not the stale one.
        await Tournament.create(id=100, name="T2")
        second = await Tournament.get(id=100)
        assert second is not first
        assert second.name == "T2"


@pytest.mark.asyncio
async def test_refresh_from_db_forces_reload(db):
    tournament = await Tournament.create(name="T1", desc="d1")
    with identity_scope():
        obj = await Tournament.get(id=tournament.id)
        obj.name = "local edit"
        await obj.refresh_from_db()
        assert obj.name == "T1"
        # The refreshed value is the new baseline: local edits are gone, so a
        # later query may refresh the field again.
        await Tournament.filter(id=tournament.id).update(name="db name")
        again = await Tournament.get(id=tournament.id)
        assert again is obj
        assert obj.name == "db name"


@pytest.mark.asyncio
async def test_nested_scopes(db):
    tournament = await Tournament.create(name="T1")
    with identity_scope():
        outer = await Tournament.get(id=tournament.id)
        with identity_scope():
            inner = await Tournament.get(id=tournament.id)
            # The inner scope sees instances registered in the outer scope.
            assert inner is outer
            eventless = await Tournament.filter(name="T1").first()
            assert eventless is outer
        # Inner scope entries are discarded on exit; outer entries survive.
        again = await Tournament.get(id=tournament.id)
        assert again is outer


@pytest.mark.asyncio
async def test_concurrent_scopes_are_isolated(db):
    tournament = await Tournament.create(name="T1")
    barrier = asyncio.Event()
    results = {}

    async def worker(name: str, new_name: str) -> None:
        with identity_scope():
            obj = await Tournament.get(id=tournament.id)
            obj.name = new_name
            # Ensure both tasks are inside their scopes at the same time.
            if name == "a":
                barrier.set()
            await barrier.wait()
            other = await Tournament.get(id=tournament.id)
            results[name] = (obj, other, obj.name)

    await asyncio.gather(worker("a", "name-a"), worker("b", "name-b"))

    obj_a, other_a, name_a = results["a"]
    obj_b, other_b, name_b = results["b"]
    # Each task saw a single shared instance within its own scope...
    assert obj_a is other_a
    assert obj_b is other_b
    # ...but the two concurrent scopes did not share instances.
    assert obj_a is not obj_b
    assert name_a == "name-a"
    assert name_b == "name-b"


@pytest.mark.asyncio
async def test_scope_is_task_local(db):
    tournament = await Tournament.create(name="T1")
    started = asyncio.Event()
    proceed = asyncio.Event()
    seen: dict = {}

    async def other_task() -> Tournament:
        started.set()
        await proceed.wait()
        seen["scope"] = current_scope()
        return await Tournament.get(id=tournament.id)

    # The other task is created before this task enters a scope, so it never
    # sees it: scopes of concurrent tasks are isolated from each other.
    task = asyncio.create_task(other_task())
    await started.wait()
    with identity_scope():
        in_scope = await Tournament.get(id=tournament.id)
        proceed.set()
        other = await task
        assert seen["scope"] is None
        assert other is not in_scope


@pytest.mark.asyncio
async def test_async_context_manager_and_clear(db):
    tournament = await Tournament.create(name="T1")
    scope = IdentityScope()
    async with scope:
        first = await Tournament.get(id=tournament.id)
        assert len(scope) == 1
        scope.clear()
        assert len(scope) == 0
        second = await Tournament.get(id=tournament.id)
        assert second is not first
    assert len(scope) == 0
