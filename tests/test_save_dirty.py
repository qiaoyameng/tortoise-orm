from asyncio import CancelledError
import re

import pytest
import pytest_asyncio

from tests.testmodels import DatetimeFields, DefaultModel, IntFields, JSONFields
from tortoise import Tortoise
from tortoise.exceptions import IncompleteInstanceError, OperationalError
from tortoise.expressions import F
from tortoise.fields.base import DatabaseDefault


def update_columns(sql: str) -> set[str]:
    match = re.fullmatch(
        r'UPDATE "([a-z_]+)" SET (.*?) WHERE "id"=\?',
        sql,
        flags=re.DOTALL,
    )
    assert match, sql
    return set(re.findall(r'"([^"]+)"(?:\s*=\s*(?!WHERE))', match.group(2)))


@pytest_asyncio.fixture
async def sqlite_executor(monkeypatch):
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["tests.testmodels"]},
    )
    connection = Tortoise.get_connection("default")
    statements: list[tuple[str, list]] = []

    async def execute_query(query, values=None):
        statements.append((query, values or []))
        return 1, []

    monkeypatch.setattr(connection, "execute_query", execute_query)
    try:
        yield statements
    finally:
        await Tortoise.close_connections()


@pytest.mark.asyncio
async def test_save_dirty_updates_only_changed_column(sqlite_executor):
    obj = IntFields._init_from_db(id=1, intnum=1, intnum_null=2)

    obj.intnum = 3
    rows = await obj.save_dirty()

    assert rows == 1
    assert len(sqlite_executor) == 1
    sql, values = sqlite_executor[0]
    assert update_columns(sql) == {"intnum"}
    assert values == [3, 1]
    assert obj.get_dirty_fields() == []


@pytest.mark.asyncio
async def test_save_dirty_ignores_value_assigned_back_to_original(sqlite_executor):
    obj = IntFields._init_from_db(id=1, intnum=1, intnum_null=2)
    original = obj.intnum_null

    obj.intnum_null = 3
    assert obj.get_dirty_fields() == ["intnum_null"]

    obj.intnum_null = original
    assert obj.get_dirty_fields() == []
    assert await obj.save_dirty() == 0
    assert sqlite_executor == []


@pytest.mark.asyncio
async def test_save_dirty_detects_json_in_place_mutation(sqlite_executor):
    obj = JSONFields._init_from_db(
        id=1,
        data={"items": []},
        data_null=None,
        data_default={"a": 1},
        data_validate=None,
        data_pydantic=None,
        data_decimal=None,
        data_index=None,
    )

    obj.data["items"].append("value")
    assert obj.get_dirty_fields() == ["data"]

    assert await obj.save_dirty() == 1
    assert update_columns(sqlite_executor[0][0]) == {"data"}


@pytest.mark.asyncio
async def test_save_dirty_json_assigned_back_to_original_value_is_clean(sqlite_executor):
    obj = JSONFields._init_from_db(
        id=1,
        data={"items": ["one"]},
        data_null=None,
        data_default={"a": 1},
        data_validate=None,
        data_pydantic=None,
        data_decimal=None,
        data_index=None,
    )

    obj.data = {"items": ["two"]}
    obj.data = {"items": ["one"]}

    assert obj.get_dirty_fields() == []
    assert await obj.save_dirty() == 0
    assert sqlite_executor == []


@pytest.mark.asyncio
async def test_save_dirty_only_considers_loaded_fields(sqlite_executor):
    partial = IntFields._init_from_db(id=1, intnum=1)
    assert partial._partial is True

    partial.intnum = 4
    assert partial.get_dirty_fields() == ["intnum"]
    assert await partial.save_dirty() == 1
    assert update_columns(sqlite_executor[0][0]) == {"intnum"}
    assert sqlite_executor[0][1] == [4, 1]


@pytest.mark.asyncio
async def test_save_dirty_only_model_without_primary_key_is_rejected(sqlite_executor):
    partial = IntFields._init_from_db(intnum=1)

    with pytest.raises(IncompleteInstanceError, match="Dirty update not available"):
        await partial.save_dirty()


@pytest.mark.asyncio
async def test_save_dirty_unsaved_model_is_rejected():
    with pytest.raises(OperationalError, match="unpersisted record"):
        await IntFields(intnum=1, intnum_null=None).save_dirty()


@pytest.mark.asyncio
async def test_save_dirty_includes_loaded_auto_now_with_other_change(sqlite_executor):
    obj = DatetimeFields._init_from_db(
        id=1,
        datetime="2024-01-01 00:00:00",
        datetime_null=None,
        datetime_auto="2024-01-01 00:00:00",
        datetime_add="2024-01-01 00:00:00",
    )
    original_auto = obj.datetime_auto

    obj.datetime_null = obj.datetime

    assert "datetime_null" in obj.get_dirty_fields()
    assert await obj.save_dirty() == 1
    assert update_columns(sqlite_executor[0][0]) == {"datetime_null", "datetime_auto"}
    assert obj.datetime_auto != original_auto
    assert obj.get_dirty_fields() == []


@pytest.mark.asyncio
async def test_save_dirty_unchanged_auto_now_does_not_update(sqlite_executor):
    obj = DatetimeFields._init_from_db(
        id=1,
        datetime="2024-01-01 00:00:00",
        datetime_null=None,
        datetime_auto="2024-01-01 00:00:00",
        datetime_add="2024-01-01 00:00:00",
    )

    assert obj.get_dirty_fields() == []
    assert await obj.save_dirty() == 0
    assert sqlite_executor == []


@pytest.mark.asyncio
async def test_save_dirty_emits_database_default_and_baselines_sentinel(
    sqlite_executor,
    monkeypatch,
):
    obj = DefaultModel._init_from_db(
        id=1,
        int_default=10,
        float_default=1.5,
        decimal_default=1,
        bool_default=True,
        char_default="tortoise",
        date_default="2024-01-01",
        datetime_default="2024-01-01 00:00:00",
    )

    obj.int_default = DefaultModel._meta.fields_map["int_default"].get_db_default_value()
    assert obj.get_dirty_fields() == ["int_default"]

    assert await obj.save_dirty() == 1
    sql, values = sqlite_executor[0]
    assert update_columns(sql) == {"int_default"}
    assert sql == 'UPDATE "defaultmodel" SET "int_default"=DEFAULT WHERE "id"=?'
    assert values == [1]
    assert isinstance(obj.int_default, DatabaseDefault)
    assert obj.get_dirty_fields() == []

    async def execute_query(query, values=None):
        return 1, [{"int_default": 1}]

    monkeypatch.setattr(Tortoise.get_connection("default"), "execute_query", execute_query)
    await obj.refresh_from_db(fields=["int_default"])
    assert obj.int_default == 1
    assert obj.get_dirty_fields() == []


@pytest.mark.asyncio
async def test_save_dirty_expression_is_clean_after_successful_write(sqlite_executor):
    obj = IntFields._init_from_db(id=1, intnum=1, intnum_null=2)

    obj.intnum = F("intnum") + 1
    assert obj.get_dirty_fields() == ["intnum"]

    assert await obj.save_dirty() == 1
    assert update_columns(sqlite_executor[0][0]) == {"intnum"}
    assert sqlite_executor[0][1] == [1]
    assert obj.get_dirty_fields() == []
    assert await obj.save_dirty() == 0
    assert len(sqlite_executor) == 1

    obj.intnum = 3
    assert obj.get_dirty_fields() == ["intnum"]
    assert await obj.save_dirty() == 1
    assert sqlite_executor[1][1] == [3, 1]
    assert obj.get_dirty_fields() == []


@pytest.mark.asyncio
async def test_save_dirty_keeps_dirty_baseline_after_failed_update(
    sqlite_executor,
    monkeypatch,
):
    obj = DatetimeFields._init_from_db(
        id=1,
        datetime="2024-01-01 00:00:00",
        datetime_null=None,
        datetime_auto="2024-01-01 00:00:00",
        datetime_add="2024-01-01 00:00:00",
    )
    original_auto = obj.datetime_auto
    obj.datetime_null = obj.datetime
    connection = Tortoise.get_connection("default")
    call_count = 0

    async def failing_execute_query(query, values=None):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(connection, "execute_query", failing_execute_query)

    with pytest.raises(RuntimeError, match="boom"):
        await obj.save_dirty()

    assert call_count == 1
    assert obj.datetime_auto == original_auto
    assert obj.get_dirty_fields() == ["datetime_null"]

    async def execute_query(query, values=None):
        sqlite_executor.append((query, values or []))
        return 1, []

    monkeypatch.setattr(connection, "execute_query", execute_query)

    assert await obj.save_dirty() == 1
    assert update_columns(sqlite_executor[0][0]) == {"datetime_null", "datetime_auto"}
    assert obj.get_dirty_fields() == []


@pytest.mark.asyncio
async def test_save_dirty_restores_baseline_on_cancellation(
    sqlite_executor,
    monkeypatch,
):
    obj = IntFields._init_from_db(id=1, intnum=1, intnum_null=2)
    obj.intnum = 3

    async def cancelled_execute_query(query, values=None):
        raise CancelledError()

    monkeypatch.setattr(
        Tortoise.get_connection("default"),
        "execute_query",
        cancelled_execute_query,
    )

    with pytest.raises(CancelledError):
        await obj.save_dirty()

    assert obj.get_dirty_fields() == ["intnum"]
    assert sqlite_executor == []


@pytest.mark.asyncio
async def test_save_dirty_zero_rows_restores_auto_now_baseline(
    sqlite_executor,
    monkeypatch,
):
    obj = DatetimeFields._init_from_db(
        id=1,
        datetime="2024-01-01 00:00:00",
        datetime_null=None,
        datetime_auto="2024-01-01 00:00:00",
        datetime_add="2024-01-01 00:00:00",
    )
    original_auto = obj.datetime_auto
    obj.datetime_null = obj.datetime

    async def execute_query(query, values=None):
        return 0, []

    monkeypatch.setattr(Tortoise.get_connection("default"), "execute_query", execute_query)

    assert await obj.save_dirty() == 0
    assert obj.datetime_auto == original_auto
    assert obj.get_dirty_fields() == ["datetime_null"]


@pytest.mark.asyncio
async def test_regular_save_still_updates_all_loaded_fields(sqlite_executor):
    obj = IntFields._init_from_db(id=1, intnum=1, intnum_null=2)

    await obj.save()

    assert len(sqlite_executor) == 1
    assert update_columns(sqlite_executor[0][0]) == {"intnum", "intnum_null"}


@pytest.mark.asyncio
async def test_regular_save_keeps_skipping_database_default_sentinel(sqlite_executor):
    obj = DefaultModel._init_from_db(
        id=1,
        int_default=10,
        float_default=1.5,
        decimal_default=1,
        bool_default=True,
        char_default="tortoise",
        date_default="2024-01-01",
        datetime_default="2024-01-01 00:00:00",
    )
    obj.int_default = DefaultModel._meta.fields_map["int_default"].get_db_default_value()

    await obj.save()

    assert len(sqlite_executor) == 1
    assert "int_default" not in sqlite_executor[0][0]
    assert "DEFAULT" not in sqlite_executor[0][0]
    assert obj.get_dirty_fields() == ["int_default"]
