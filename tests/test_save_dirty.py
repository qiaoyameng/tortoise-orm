"""
Tests for instance dirty-field tracking and ``Model.save_dirty()``.

The UPDATE statements emitted against SQLite are captured through the
``tortoise.db_client`` logger so the exact updated column sets can be
asserted.
"""

import asyncio
import logging
import re

import pytest

from tests.testmodels import (
    DatetimeFields,
    DefaultModel,
    Event,
    IntFields,
    JSONFields,
    Tournament,
)
from tortoise.exceptions import OperationalError
from tortoise.expressions import F

UPDATE_RE = re.compile(r'UPDATE "[^"]+" SET (?P<sets>.+?) WHERE')


def update_column_sets(caplog):
    """Return the list of updated column sets for every captured UPDATE."""
    statements = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("UPDATE"):
            continue
        match = UPDATE_RE.search(message)
        assert match, f"unparseable UPDATE statement: {message}"
        columns = {
            assignment.split("=")[0].strip().strip('"')
            for assignment in match.group("sets").split(",")
        }
        statements.append(columns)
    return statements


@pytest.fixture
def update_log(caplog):
    """Capture SQL statements logged by the db client."""
    with caplog.at_level(logging.DEBUG, logger="tortoise.db_client"):
        yield caplog


@pytest.mark.asyncio
async def test_save_dirty_updates_only_changed_column(db, update_log):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.get(pk=tournament.pk)

    obj.name = "b"
    updated = await obj.save_dirty()

    assert updated == ["name"]
    assert update_column_sets(update_log) == [{"name"}]
    await obj.refresh_from_db()
    assert obj.name == "b"
    assert obj.desc == "desc"


@pytest.mark.asyncio
async def test_save_dirty_without_changes_emits_no_update(db, update_log):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.get(pk=tournament.pk)

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_save_dirty_value_reverted_to_original_is_clean(db, update_log):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.get(pk=tournament.pk)

    obj.name = "b"
    obj.name = "a"  # assigned back to the loaded value

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_save_dirty_on_unsaved_instance_inserts(db, update_log):
    obj = Tournament(name="new", desc="desc")

    assert await obj.save_dirty() is None
    assert obj.pk is not None

    obj.desc = "changed"
    assert await obj.save_dirty() == ["desc"]
    assert update_column_sets(update_log) == [{"desc"}]


@pytest.mark.asyncio
async def test_save_dirty_partial_model_only_updates_loaded_fields(db, update_log):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.all().only("id", "name").get(pk=tournament.pk)

    obj.name = "b"
    updated = await obj.save_dirty()

    assert updated == ["name"]
    assert update_column_sets(update_log) == [{"name"}]
    await tournament.refresh_from_db()
    assert tournament.name == "b"
    assert tournament.desc == "desc"


@pytest.mark.asyncio
async def test_save_dirty_partial_model_without_changes_emits_no_update(db, update_log):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.all().only("id", "name").get(pk=tournament.pk)

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_save_dirty_detects_json_in_place_mutation(db, update_log):
    obj = await JSONFields.create(data={"a": 1})
    obj = await JSONFields.get(pk=obj.pk)

    obj.data["b"] = 2
    updated = await obj.save_dirty()

    assert updated == ["data"]
    assert update_column_sets(update_log) == [{"data"}]
    await obj.refresh_from_db()
    assert obj.data == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_save_dirty_json_reassigned_to_equal_value_is_clean(db, update_log):
    obj = await JSONFields.create(data={"a": 1})
    obj = await JSONFields.get(pk=obj.pk)

    obj.data = {"a": 1}

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_save_dirty_failed_update_keeps_fields_dirty(db, update_log, monkeypatch):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.get(pk=tournament.pk)
    obj.name = "b"

    executor_class = Tournament._meta.db.executor_class
    original_execute_update = executor_class.execute_update

    async def failing_execute_update(self, instance, update_fields):
        raise OperationalError("simulated database failure")

    monkeypatch.setattr(executor_class, "execute_update", failing_execute_update)
    with pytest.raises(OperationalError):
        await obj.save_dirty()
    monkeypatch.setattr(executor_class, "execute_update", original_execute_update)

    # The failed write must not have marked the field as clean.
    assert obj._get_dirty_fields() == {"name"}
    assert await obj.save_dirty() == ["name"]
    assert update_column_sets(update_log) == [{"name"}]
    # And after the successful retry the baseline is clean again.
    assert await obj.save_dirty() == []


@pytest.mark.asyncio
async def test_save_dirty_cancelled_update_keeps_fields_dirty(db, update_log, monkeypatch):
    tournament = await Tournament.create(name="a", desc="desc")
    obj = await Tournament.get(pk=tournament.pk)
    obj.name = "b"

    executor_class = Tournament._meta.db.executor_class

    async def cancelling_execute_update(self, instance, update_fields):
        raise asyncio.CancelledError

    monkeypatch.setattr(executor_class, "execute_update", cancelling_execute_update)
    with pytest.raises(asyncio.CancelledError):
        await obj.save_dirty()
    monkeypatch.undo()

    assert obj._get_dirty_fields() == {"name"}
    assert await obj.save_dirty() == ["name"]


@pytest.mark.asyncio
async def test_save_dirty_expression_assignment(db, update_log):
    obj = await IntFields.create(intnum=10)
    obj = await IntFields.get(pk=obj.pk)

    obj.intnum = F("intnum") + 5
    updated = await obj.save_dirty()

    assert updated == ["intnum"]
    assert update_column_sets(update_log) == [{"intnum"}]

    # The resulting DB value is unknown to the instance, so the field stays
    # dirty until it is refreshed from the database.
    assert "intnum" in obj._get_dirty_fields()
    await obj.refresh_from_db()
    assert obj.intnum == 15
    assert obj._get_dirty_fields() == set()


@pytest.mark.asyncio
async def test_save_dirty_includes_auto_now_fields(db, update_log):
    obj = await DatetimeFields.create(datetime="2021-01-01 00:00:00")
    obj = await DatetimeFields.get(pk=obj.pk)

    obj.datetime_null = obj.datetime
    updated = await obj.save_dirty()

    assert updated == ["datetime_auto", "datetime_null"]
    assert update_column_sets(update_log) == [{"datetime_auto", "datetime_null"}]


@pytest.mark.asyncio
async def test_save_dirty_auto_now_alone_does_not_trigger_update(db, update_log):
    obj = await DatetimeFields.create(datetime="2021-01-01 00:00:00")
    obj = await DatetimeFields.get(pk=obj.pk)

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_save_dirty_fk_assignment(db, update_log):
    tournament_a = await Tournament.create(name="a")
    tournament_b = await Tournament.create(name="b")
    event = await Event.create(name="e", tournament=tournament_a)
    obj = await Event.get(pk=event.pk)

    obj.tournament = tournament_b
    updated = await obj.save_dirty()

    assert updated == ["modified", "tournament_id"]
    assert update_column_sets(update_log) == [{"modified", "tournament_id"}]
    await obj.refresh_from_db()
    assert obj.tournament_id == tournament_b.pk


@pytest.mark.asyncio
async def test_save_dirty_db_default_fields_are_clean(db, update_log):
    obj = await DefaultModel.create()

    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []

    obj.int_default = 42
    assert await obj.save_dirty() == ["int_default"]
    assert update_column_sets(update_log) == [{"int_default"}]


@pytest.mark.asyncio
async def test_bulk_update_advances_baseline(db, update_log):
    objs = [await Tournament.create(name="a"), await Tournament.create(name="b")]
    objs[0].name = "x"
    objs[1].name = "y"
    await Tournament.bulk_update(objs, fields=["name"])

    update_log.records.clear()
    assert await objs[0].save_dirty() == []
    assert await objs[1].save_dirty() == []
    assert update_column_sets(update_log) == []


@pytest.mark.asyncio
async def test_refresh_from_db_resets_baseline(db, update_log):
    tournament = await Tournament.create(name="a")
    obj = await Tournament.get(pk=tournament.pk)

    obj.name = "b"
    await obj.refresh_from_db()

    assert obj.name == "a"
    assert await obj.save_dirty() == []
    assert update_column_sets(update_log) == []
