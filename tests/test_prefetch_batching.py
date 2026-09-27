"""
Tests for batched prefetching.

Prefetch queries put the parent keys in an ``IN`` lookup, which can exceed the
backend parameter limit when many parent objects are fetched at once. The
executor splits the parent keys into batches that fit the parameter budget
(``Capabilities.max_query_params``, optionally tightened with
``prefetch_related(..., batch_size=...)``) and merges the results.
"""

import pytest

from tests.testmodels import Address, Event, Team, Tournament
from tortoise import connections
from tortoise.backends.base.client import Capabilities
from tortoise.exceptions import OperationalError
from tortoise.query_utils import Prefetch


def _capabilities_with_budget(caps: Capabilities, budget: int | None) -> Capabilities:
    kwargs = {
        key: value
        for key, value in caps.__dict__.items()
        if not key.startswith("_") and key != "max_query_params"
    }
    dialect = kwargs.pop("dialect")
    return Capabilities(dialect, max_query_params=budget, **kwargs)


@pytest.fixture
def query_log(db, monkeypatch):
    """Record (sql, params) of every query executed on the models connection."""
    conn = connections.get("models")
    log: list[tuple[str, list]] = []
    original = type(conn).execute_query

    async def execute_query(self, query, values=None):
        log.append((query, list(values or [])))
        return await original(self, query, values)

    monkeypatch.setattr(type(conn), "execute_query", execute_query)
    return log


@pytest.fixture
def set_param_budget(db, monkeypatch):
    """Override the backend parameter budget for the models connection."""
    conn = connections.get("models")

    def set_budget(budget: int | None) -> None:
        monkeypatch.setattr(
            conn, "capabilities", _capabilities_with_budget(conn.capabilities, budget)
        )

    return set_budget


def _in_queries(query_log):
    return [(sql, params) for sql, params in query_log if " IN (" in sql.upper()]


async def _make_tournaments_with_events(count: int, events_per_tournament: int = 2) -> None:
    for i in range(count):
        tournament = await Tournament.create(name=f"Tournament{i}")
        for j in range(events_per_tournament):
            await Event.create(name=f"Event{i}-{j}", tournament=tournament)


def _events_graph(tournaments) -> dict:
    return {t.name: [e.name for e in t.events] for t in tournaments}


@pytest.mark.asyncio
async def test_reverse_fk_split_into_batches(db, query_log, set_param_budget):
    await _make_tournaments_with_events(7)
    reference = _events_graph(await Tournament.all().order_by("id").prefetch_related("events"))

    query_log.clear()
    set_param_budget(2)  # room for 2 parent keys per query -> 4 batches
    tournaments = await Tournament.all().order_by("id").prefetch_related("events")

    in_queries = _in_queries(query_log)
    assert len(in_queries) >= 3
    assert all(len(params) <= 2 for _, params in in_queries)
    assert [len(params) for _, params in in_queries] == [2, 2, 2, 1]
    assert _events_graph(tournaments) == reference


@pytest.mark.asyncio
async def test_m2m_split_into_batches(db, query_log, set_param_budget):
    teams = [await Team.create(name=f"Team{i}") for i in range(3)]
    for i in range(7):
        tournament = await Tournament.create(name=f"Tournament{i}")
        event = await Event.create(name=f"Event{i}", tournament=tournament)
        await event.participants.add(*teams[: i % 3 + 1])

    async def graph():
        events = await Event.all().order_by("event_id").prefetch_related("participants")
        return {e.name: [t.name for t in e.participants] for e in events}

    reference = await graph()
    query_log.clear()
    set_param_budget(2)  # room for 2 parent keys per query -> 4 batches
    batched = await graph()

    in_queries = _in_queries(query_log)
    assert len(in_queries) >= 3
    assert all(len(params) <= 2 for _, params in in_queries)
    assert batched == reference


@pytest.mark.asyncio
async def test_reverse_o2o_split_into_batches(db, query_log, set_param_budget):
    tournament = await Tournament.create(name="Tournament")
    for i in range(5):
        event = await Event.create(name=f"Event{i}", tournament=tournament)
        await Address.create(city=f"City{i}", street=f"Street{i}", event=event)

    query_log.clear()
    set_param_budget(2)  # room for 2 parent keys per query -> 3 batches
    events = await Event.all().order_by("event_id").prefetch_related("address")

    in_queries = _in_queries(query_log)
    assert len(in_queries) >= 3
    assert all(len(params) <= 2 for _, params in in_queries)
    assert [e.address.city for e in events] == [f"City{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_direct_relation_dedupes_parent_keys(db, query_log):
    tournament1 = await Tournament.create(name="Tournament1")
    tournament2 = await Tournament.create(name="Tournament2")
    for i in range(6):
        await Event.create(name=f"Event{i}", tournament=tournament1 if i % 2 else tournament2)

    query_log.clear()
    events = await Event.all().order_by("event_id").prefetch_related("tournament")

    in_queries = _in_queries(query_log)
    # 6 parent objects but only 2 distinct parent keys -> a single query
    assert len(in_queries) == 1
    assert len(in_queries[0][1]) == 2
    assert [e.tournament.name for e in events] == [
        "Tournament2" if i % 2 == 0 else "Tournament1" for i in range(6)
    ]


@pytest.mark.asyncio
async def test_batch_size_config(db, query_log):
    await _make_tournaments_with_events(5)

    query_log.clear()
    tournaments = await Tournament.all().order_by("id").prefetch_related("events", batch_size=2)

    in_queries = _in_queries(query_log)
    assert [len(params) for _, params in in_queries] == [2, 2, 1]
    assert _events_graph(tournaments) == {
        f"Tournament{i}": [f"Event{i}-0", f"Event{i}-1"] for i in range(5)
    }


@pytest.mark.asyncio
async def test_batch_size_config_must_be_positive(db):
    with pytest.raises(ValueError):
        await Tournament.all().prefetch_related("events", batch_size=0)


@pytest.mark.asyncio
async def test_no_parent_objects(db, query_log):
    await _make_tournaments_with_events(2)

    query_log.clear()
    tournaments = await Tournament.filter(name="no-such-tournament").prefetch_related("events")

    assert tournaments == []
    assert _in_queries(query_log) == []


@pytest.mark.asyncio
async def test_fixed_params_consume_whole_budget(db, query_log, set_param_budget):
    await _make_tournaments_with_events(3)

    query_log.clear()
    # The related query itself uses 1 parameter (the icontains pattern), which
    # already eats the whole budget of 1 -> fall back to one parent key per query.
    set_param_budget(1)
    tournaments = await Tournament.all().order_by("id").prefetch_related(
        Prefetch("events", queryset=Event.filter(name__icontains="event"))
    )

    in_queries = _in_queries(query_log)
    assert len(in_queries) == 3
    # 1 fixed parameter + 1 parent key per query
    assert all(len(params) == 2 for _, params in in_queries)
    assert _events_graph(tournaments) == {
        f"Tournament{i}": [f"Event{i}-0", f"Event{i}-1"] for i in range(3)
    }


@pytest.mark.asyncio
async def test_failed_batch_leaves_no_partial_cache(db, monkeypatch, set_param_budget):
    await _make_tournaments_with_events(5)
    tournaments = await Tournament.all().order_by("id")

    set_param_budget(2)  # 3 batches
    conn = connections.get("models")
    original = type(conn).execute_query
    in_query_count = 0

    async def failing_execute_query(self, query, values=None):
        nonlocal in_query_count
        if " IN (" in query.upper():
            in_query_count += 1
            if in_query_count == 2:
                raise OperationalError("simulated batch failure")
        return await original(self, query, values)

    monkeypatch.setattr(type(conn), "execute_query", failing_execute_query)

    with pytest.raises(OperationalError, match="simulated batch failure"):
        await Tournament.fetch_for_list(tournaments, "events")

    assert in_query_count == 2
    # No instance may expose a partially populated relation cache.
    for tournament in tournaments:
        assert tournament.events._fetched is False
        assert tournament.events.related_objects == []


@pytest.mark.asyncio
async def test_to_attr_and_ordering_preserved(db, query_log, set_param_budget):
    for i in range(5):
        tournament = await Tournament.create(name=f"Tournament{i}")
        # create events in non-alphabetical order to make ordering observable
        for name in ("b", "a", "c"):
            await Event.create(name=f"{name}{i}", tournament=tournament)

    set_param_budget(2)
    tournaments = await Tournament.all().order_by("id").prefetch_related(
        Prefetch("events", queryset=Event.all().order_by("-name"), to_attr="sorted_events")
    )

    in_queries = _in_queries(query_log)
    assert len(in_queries) >= 3
    for i, tournament in enumerate(tournaments):
        assert [e.name for e in tournament.sorted_events] == [f"c{i}", f"b{i}", f"a{i}"]


@pytest.mark.asyncio
async def test_nested_prefetch_preserved(db, query_log, set_param_budget):
    teams = [await Team.create(name=f"Team{i}") for i in range(2)]
    for i in range(5):
        tournament = await Tournament.create(name=f"Tournament{i}")
        for j in range(2):
            event = await Event.create(name=f"Event{i}-{j}", tournament=tournament)
            await event.participants.add(*teams[: (i + j) % 2 + 1])

    async def graph():
        tournaments = await Tournament.all().order_by("id").prefetch_related(
            "events__participants"
        )
        return {
            t.name: {e.name: [team.name for team in e.participants] for e in t.events}
            for t in tournaments
        }

    reference = await graph()
    query_log.clear()
    set_param_budget(2)
    batched = await graph()

    in_queries = _in_queries(query_log)
    assert len(in_queries) >= 3
    assert all(len(params) <= 2 for _, params in in_queries)
    assert batched == reference
