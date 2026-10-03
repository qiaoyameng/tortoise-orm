import asyncio
import datetime
from decimal import Decimal

import aiosqlite
import pytest
import pytest_asyncio

from tests.testmodels import DecimalFields, IntFields, TimeDeltaFields, Tournament
from tortoise.backends.base.client import Capabilities
from tortoise.contrib.test import requireCapability
from tortoise.exceptions import ParamsError, UnSupportedError
from tortoise.transactions import in_transaction


@pytest_asyncio.fixture
async def intfields_rows(db):
    for val in range(25):
        await IntFields.create(intnum=val)
    return 25


class CursorRecorder:
    """Records fetchmany() sizes and close() calls on aiosqlite cursors."""

    def __init__(self, monkeypatch):
        self.fetched: list[int] = []
        self.closed = 0
        recorder = self

        orig_fetchmany = aiosqlite.Cursor.fetchmany
        orig_close = aiosqlite.Cursor.close

        async def spy_fetchmany(self, size=None):
            rows = await orig_fetchmany(self, size)
            recorder.fetched.append(len(rows))
            return rows

        async def spy_close(self):
            recorder.closed += 1
            await orig_close(self)

        monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", spy_fetchmany)
        monkeypatch.setattr(aiosqlite.Cursor, "close", spy_close)

    @property
    def total_fetched(self) -> int:
        return sum(self.fetched)


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_models(db, intfields_rows):
    seen = [obj.intnum async for obj in IntFields.all().order_by("id").iterator(chunk_size=7)]
    assert seen == list(range(25))


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_values(db):
    await DecimalFields.create(decimal=Decimal("1.5"), decimal_nodec=2)
    expected = await DecimalFields.all().values("decimal", "decimal_nodec")
    streamed = [
        row
        async for row in DecimalFields.all()
        .values("decimal", "decimal_nodec")
        .iterator(chunk_size=1)
    ]
    assert streamed == expected
    assert isinstance(streamed[0]["decimal"], Decimal)


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_values_list(db, intfields_rows):
    queryset = IntFields.all().order_by("id").values_list("id", "intnum")
    expected = await queryset
    streamed = [row async for row in queryset.iterator(chunk_size=4)]
    assert streamed == expected


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_values_list_flat(db, intfields_rows):
    queryset = IntFields.all().order_by("id").values_list("intnum", flat=True)
    expected = await queryset
    streamed = [val async for val in queryset.iterator(chunk_size=4)]
    assert streamed == expected


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_full_consumption_reads_whole_table(db, intfields_rows, monkeypatch):
    recorder = CursorRecorder(monkeypatch)
    seen = [obj.intnum async for obj in IntFields.all().order_by("id").iterator(chunk_size=7)]
    assert seen == list(range(25))
    # 7 + 7 + 7 + 4 rows, then an empty fetch signalling exhaustion
    assert recorder.fetched == [7, 7, 7, 4, 0]
    assert recorder.closed == 1


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_early_close_does_not_read_whole_table(
    db, intfields_rows, monkeypatch
):
    recorder = CursorRecorder(monkeypatch)
    iterator = IntFields.all().order_by("id").iterator(chunk_size=5)
    first = await iterator.__anext__()
    second = await iterator.__anext__()
    assert (first.intnum, second.intnum) == (0, 1)
    await iterator.aclose()

    # Only one chunk of 5 rows was fetched: the table was NOT read in full,
    # and the cursor was released.
    assert recorder.total_fetched == 5
    assert recorder.total_fetched < intfields_rows
    assert recorder.closed == 1

    # The same connection is immediately usable for further queries.
    assert await IntFields.all().count() == 25


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_cancellation_releases_cursor(db, intfields_rows, monkeypatch):
    closed = 0
    orig_close = aiosqlite.Cursor.close

    async def spy_close(self):
        nonlocal closed
        closed += 1
        await orig_close(self)

    monkeypatch.setattr(aiosqlite.Cursor, "close", spy_close)

    fetch_started = asyncio.Event()
    orig_fetchmany = aiosqlite.Cursor.fetchmany

    async def blocking_fetchmany(self, size=None):
        rows = await orig_fetchmany(self, size)
        fetch_started.set()
        await asyncio.sleep(60)
        return rows

    monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", blocking_fetchmany)

    async def consume():
        async for _ in IntFields.all().iterator(chunk_size=5):
            pass

    task = asyncio.create_task(consume())
    await fetch_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == 1
    assert await IntFields.all().count() == 25


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_conversion_error_releases_cursor(db, monkeypatch):
    await TimeDeltaFields.create(timedelta=datetime.timedelta(days=1))
    await TimeDeltaFields.create(timedelta=datetime.timedelta(days=2))
    closed = 0
    orig_close = aiosqlite.Cursor.close

    async def spy_close(self):
        nonlocal closed
        closed += 1
        await orig_close(self)

    monkeypatch.setattr(aiosqlite.Cursor, "close", spy_close)

    def broken_to_python(value):
        raise ValueError("cannot convert")

    field = TimeDeltaFields._meta.fields_map["timedelta"]
    monkeypatch.setattr(field, "to_python_value", broken_to_python)

    with pytest.raises(ValueError, match="cannot convert"):
        async for _ in TimeDeltaFields.all().values_list("timedelta").iterator(chunk_size=5):
            pass

    assert closed == 1
    assert await TimeDeltaFields.all().count() == 2


@pytest.mark.asyncio
@requireCapability(support_streaming=True)
async def test_iterator_inside_transaction_uses_same_connection(db_truncate, monkeypatch):
    async with in_transaction() as conn:
        await Tournament.create(name="inside-tx")
        used = []
        orig_stream = conn.execute_query_stream

        def spy(query, values=None, chunk_size=2000):
            used.append(conn)
            return orig_stream(query, values, chunk_size)

        monkeypatch.setattr(conn, "execute_query_stream", spy)

        names = [t.name async for t in Tournament.all().iterator(chunk_size=2)]
        # The uncommitted row is visible, and the stream ran on the
        # transaction's own connection object.
        assert "inside-tx" in names
        assert used == [conn]


@pytest.mark.asyncio
async def test_iterator_prefetch_rejected_before_execution(db):
    queryset = Tournament.all().prefetch_related("events")
    with pytest.raises(UnSupportedError, match="prefetch"):
        queryset.iterator()


@pytest.mark.asyncio
async def test_iterator_single_object_rejected(db):
    with pytest.raises(UnSupportedError):
        IntFields.all().first().iterator()


@pytest.mark.asyncio
async def test_iterator_unsupported_backend_rejected(db, monkeypatch):
    connection = IntFields._meta.db
    monkeypatch.setattr(connection, "capabilities", Capabilities("sqlite"))
    with pytest.raises(UnSupportedError, match="streaming"):
        IntFields.all().iterator()


@pytest.mark.asyncio
async def test_iterator_invalid_chunk_size(db):
    with pytest.raises(ParamsError):
        IntFields.all().iterator(chunk_size=0)
    with pytest.raises(ParamsError):
        IntFields.all().values_list("id").iterator(chunk_size=-1)
