"""
Tests for QuerySet.iterator() cursor-based streaming.
"""

import asyncio
import datetime

import aiosqlite
import pytest
import pytest_asyncio

from tests.testmodels import DatetimeFields, Event, IntFields, Tournament
from tortoise import connections
from tortoise.backends.base.client import BaseDBAsyncClient, Capabilities
from tortoise.backends.sqlite.client import SqliteClient, SqliteClientWithRegexpSupport
from tortoise.contrib.test import requireCapability
from tortoise.exceptions import ParamsError, UnSupportedError
from tortoise.expressions import F
from tortoise.transactions import in_transaction

ROW_COUNT = 25
CHUNK_SIZE = 10


@pytest_asyncio.fixture
async def intfields_rows(db):
    for val in range(ROW_COUNT):
        await IntFields.create(intnum=val)
    return db


class FetchmanyRecorder:
    """Records aiosqlite cursor fetchmany() batch sizes and close() calls."""

    def __init__(self, monkeypatch):
        self.batches: list[int] = []
        self.closed: list[aiosqlite.Cursor] = []
        original_fetchmany = aiosqlite.Cursor.fetchmany
        original_close = aiosqlite.Cursor.close

        async def fetchmany(cursor, size=None):
            rows = await original_fetchmany(cursor, size)
            self.batches.append(len(rows))
            return rows

        async def close(cursor):
            self.closed.append(cursor)
            await original_close(cursor)

        monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", fetchmany)
        monkeypatch.setattr(aiosqlite.Cursor, "close", close)


@pytest.fixture
def fetchmany_recorder(monkeypatch):
    return FetchmanyRecorder(monkeypatch)


def test_capability_declared_per_backend():
    assert Capabilities("sqlite").support_cursor_iteration is False
    assert SqliteClient.capabilities.support_cursor_iteration is True
    assert SqliteClientWithRegexpSupport.capabilities.support_cursor_iteration is True


def test_base_client_does_not_emulate_cursor_iteration():
    client = BaseDBAsyncClient(connection_name="fake")
    with pytest.raises(NotImplementedError):
        client.execute_query_cursor("SELECT 1")


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_models_reads_in_batches(intfields_rows, fetchmany_recorder):
    seen = [
        obj.intnum
        async for obj in IntFields.all().order_by("intnum").iterator(chunk_size=CHUNK_SIZE)
    ]
    assert seen == list(range(ROW_COUNT))
    # rows were fetched incrementally in chunk_size batches, not via fetchall
    assert fetchmany_recorder.batches == [CHUNK_SIZE, CHUNK_SIZE, 5, 0]
    assert len(fetchmany_recorder.closed) == 1


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_partial_read_releases_cursor_and_connection(
    intfields_rows, fetchmany_recorder
):
    iterator = IntFields.all().order_by("intnum").iterator(chunk_size=CHUNK_SIZE)
    first = await anext(iterator)
    assert first.intnum == 0
    await iterator.aclose()

    # only a single batch was read: the whole table was NOT consumed
    assert sum(fetchmany_recorder.batches) == CHUNK_SIZE < ROW_COUNT
    # the cursor owned by the iterator was released
    assert len(fetchmany_recorder.closed) == 1

    # the same connection immediately continues to serve queries
    assert await IntFields.all().count() == ROW_COUNT
    obj = await IntFields.create(intnum=1000)
    assert obj.id is not None


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_exhausts_then_connection_reusable(intfields_rows, fetchmany_recorder):
    seen = [obj async for obj in IntFields.all().iterator(chunk_size=CHUNK_SIZE)]
    assert len(seen) == ROW_COUNT
    assert len(fetchmany_recorder.closed) == 1
    assert await IntFields.all().count() == ROW_COUNT


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_values_applies_field_conversion(db, fetchmany_recorder):
    await DatetimeFields.create(datetime=datetime.datetime(2024, 1, 2, 3, 4, 5))
    expected = await DatetimeFields.all().values("datetime")
    seen = [row async for row in DatetimeFields.all().values("datetime").iterator(chunk_size=1)]
    assert seen == expected
    assert isinstance(seen[0]["datetime"], datetime.datetime)
    assert len(fetchmany_recorder.closed) == 1


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_values_list(intfields_rows):
    expected = await IntFields.all().order_by("intnum").values_list("id", "intnum")
    seen = [
        row
        async for row in IntFields.all()
        .order_by("intnum")
        .values_list("id", "intnum")
        .iterator(chunk_size=7)
    ]
    assert seen == expected


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_values_list_flat(intfields_rows):
    seen = [
        val
        async for val in IntFields.all()
        .order_by("intnum")
        .values_list("intnum", flat=True)
        .iterator(chunk_size=7)
    ]
    assert seen == list(range(ROW_COUNT))


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_with_annotation(intfields_rows):
    seen = [
        (obj.intnum, obj.doubled)
        async for obj in IntFields.all()
        .annotate(doubled=F("intnum") * 2)
        .order_by("intnum")
        .iterator(chunk_size=7)
    ]
    assert seen == [(val, val * 2) for val in range(ROW_COUNT)]


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_with_limit_offset(intfields_rows):
    seen = [
        obj.intnum
        async for obj in IntFields.all().order_by("intnum").offset(5).limit(7).iterator(chunk_size=3)
    ]
    assert seen == list(range(5, 12))


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_queryset_remains_awaitable(intfields_rows):
    queryset = IntFields.all().order_by("intnum")
    seen = [obj.intnum async for obj in queryset.iterator(chunk_size=7)]
    assert seen == list(range(ROW_COUNT))
    # building the iterator did not break the queryset itself
    assert [obj.intnum for obj in await queryset] == list(range(ROW_COUNT))


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_cancellation_releases_cursor(intfields_rows, monkeypatch):
    closed = []
    gate = asyncio.Event()
    original_fetchmany = aiosqlite.Cursor.fetchmany
    original_close = aiosqlite.Cursor.close
    calls = 0

    async def gated_fetchmany(cursor, size=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            # block the second batch fetch so cancellation lands mid-iteration
            await gate.wait()
        return await original_fetchmany(cursor, size)

    async def close(cursor):
        closed.append(cursor)
        await original_close(cursor)

    monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", gated_fetchmany)
    monkeypatch.setattr(aiosqlite.Cursor, "close", close)

    async def consume():
        return [obj async for obj in IntFields.all().iterator(chunk_size=5)]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    assert calls == 2  # waiting on the second batch
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()

    assert len(closed) == 1
    assert await IntFields.all().count() == ROW_COUNT


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_conversion_error_releases_cursor(db, fetchmany_recorder):
    await DatetimeFields.create(datetime=datetime.datetime(2024, 1, 1))
    conn = connections.get("models")
    await conn.execute_query("UPDATE datetimefields SET datetime='not-a-date'")

    with pytest.raises(ValueError):
        async for _ in DatetimeFields.all().values("datetime").iterator(chunk_size=5):
            pass

    assert len(fetchmany_recorder.closed) == 1
    # the connection is still usable after the failed iteration
    assert await DatetimeFields.all().count() == 1


@requireCapability(support_cursor_iteration=True)
@pytest.mark.asyncio
async def test_iterator_inside_transaction_uses_same_connection(db):
    async with in_transaction() as conn:
        await IntFields.create(intnum=42)
        # sqlite :memory: is per-connection: seeing the uncommitted row proves
        # the iterator reads from the transaction connection
        seen = [obj.intnum async for obj in IntFields.all().iterator(chunk_size=2)]
        assert seen == [42]
        assert connections.get("models")._connection is conn._connection


@pytest.mark.asyncio
async def test_iterator_rejects_unsupported_queries_before_execution(db):
    with pytest.raises(ParamsError, match="prefetch_related"):
        Tournament.all().prefetch_related("events").iterator()
    with pytest.raises(ParamsError, match="select_related"):
        Event.all().select_related("tournament").iterator()
    with pytest.raises(ParamsError, match="select_for_update"):
        # select_for_update() is a no-op on sqlite, so set the flag directly
        queryset = Tournament.all()
        queryset._select_for_update = True
        queryset.iterator()
    with pytest.raises(ParamsError, match="single-object"):
        Tournament.get(id=1).iterator()
    with pytest.raises(ParamsError, match="chunk_size"):
        Tournament.all().iterator(chunk_size=0)
    with pytest.raises(ParamsError, match="chunk_size"):
        Tournament.all().iterator(chunk_size=True)


@requireCapability(dialect="sqlite")
@pytest.mark.asyncio
async def test_iterator_raises_for_backend_without_capability(db, monkeypatch):
    conn = connections.get("models")
    monkeypatch.setattr(conn, "capabilities", Capabilities("sqlite"))
    with pytest.raises(UnSupportedError):
        IntFields.all().iterator()
    with pytest.raises(UnSupportedError):
        IntFields.all().values("intnum").iterator()
    with pytest.raises(UnSupportedError):
        IntFields.all().values_list("intnum").iterator()
