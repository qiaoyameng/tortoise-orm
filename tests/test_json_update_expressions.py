import pytest
from pypika_tortoise import Dialects
from pypika_tortoise.context import DEFAULT_SQL_CONTEXT
from pypika_tortoise.terms import Field as PypikaField

from tests.testmodels import IntFields, JSONFields
from tortoise.exceptions import FieldError, OperationalError
from tortoise.expressions import JSONRemove, JSONSet, _JSONUpdateTerm


@pytest.mark.asyncio
async def test_json_set_basic(db):
    obj = await JSONFields.create(data={"a": {"b": 1}, "c": [1, 2]})
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["a", "b"], 2)))
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"a": {"b": 2}, "c": [1, 2]}


@pytest.mark.asyncio
async def test_json_set_preserves_json_types(db):
    obj = await JSONFields.create(data={})
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONSet(
            "data",
            (["flag"], True),
            (["off"], False),
            (["num"], 1.5),
            (["nothing"], None),
            (["text"], "hi"),
            (["nested"], {"x": [1, 2]}),
        )
    )
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {
        "flag": True,
        "off": False,
        "num": 1.5,
        "nothing": None,
        "text": "hi",
        "nested": {"x": [1, 2]},
    }
    assert obj2.data["flag"] is True
    assert obj2.data["off"] is False
    assert obj2.data["nothing"] is None


@pytest.mark.asyncio
async def test_json_set_keys_with_dots_and_quotes(db):
    obj = await JSONFields.create(data={})
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONSet(
            "data",
            (["a.b"], 1),
            (["it's"], 2),
            (['say "hi"'], 3),
            (["back\\slash"], 4),
            (["nested", "a.b", 0], 5),
        )
    )
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {
        "a.b": 1,
        "it's": 2,
        'say "hi"': 3,
        "back\\slash": 4,
        "nested": {"a.b": [5]},
    }


@pytest.mark.asyncio
async def test_json_set_creates_missing_parents(db):
    obj = await JSONFields.create(data={})
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["a", "b", "c"], 1)))
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"a": {"b": {"c": 1}}}


@pytest.mark.asyncio
async def test_json_set_array_out_of_bounds_is_noop(db):
    obj = await JSONFields.create(data={"arr": [1, 2]})
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["arr", 5], 9)))
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"arr": [1, 2]}
    # In-bounds index still updates
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["arr", 1], 9)))
    obj3 = await JSONFields.get(pk=obj.pk)
    assert obj3.data == {"arr": [1, 9]}


@pytest.mark.asyncio
async def test_json_set_sql_null_is_noop(db):
    obj = await JSONFields.create(data={}, data_null=None)
    await JSONFields.filter(pk=obj.pk).update(data_null=JSONSet("data_null", (["a"], 1)))
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data_null is None


@pytest.mark.asyncio
async def test_json_set_multi_op_ordering(db):
    obj = await JSONFields.create(data={"a": {"old": True}})
    # Later operations observe the results of earlier ones.
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", (["a"], {}), (["a", "b"], 2), (["a", "b"], 3))
    )
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"a": {"b": 3}}


@pytest.mark.asyncio
async def test_json_remove(db):
    obj = await JSONFields.create(data={"a": 1, "b": {"c": 2, "d": 3}, "arr": [1, 2, 3]})
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONRemove("data", ["a"], ["b", "c"], ["arr", 0])
    )
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"b": {"d": 3}, "arr": [2, 3]}


@pytest.mark.asyncio
async def test_json_remove_missing_path_is_noop(db):
    obj = await JSONFields.create(data={"a": 1, "arr": [1, 2]})
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONRemove("data", ["missing"], ["arr", 9], ["a", "deeper"])
    )
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"a": 1, "arr": [1, 2]}


@pytest.mark.asyncio
async def test_two_updates_on_different_subfields_both_persist(db):
    obj = await JSONFields.create(data={"x": 1, "y": 1})
    # Two separate updates, each touching a different subfield; neither may
    # clobber the other (no read-modify-write cycle involved).
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["x"], 100)))
    await JSONFields.filter(pk=obj.pk).update(data=JSONSet("data", (["y"], 200)))
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"x": 100, "y": 200}


@pytest.mark.asyncio
async def test_malicious_path_cannot_change_sql_structure(db):
    obj = await JSONFields.create(data={"safe": 1})
    evil_key = "']; DROP TABLE jsonfields; --"
    await JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ([evil_key], "pwned"), (["a.b", "c"], 2))
    )
    # Table still exists, and the malicious text became a literal key.
    obj2 = await JSONFields.get(pk=obj.pk)
    assert obj2.data == {"safe": 1, evil_key: "pwned", "a.b": {"c": 2}}
    assert await JSONFields.all().count() == 1


@pytest.mark.asyncio
async def test_json_set_rejects_non_json_field(db):
    await IntFields.create(intnum=1)
    with pytest.raises(FieldError):
        await IntFields.filter().update(intnum=JSONSet("intnum", (["a"], 1)))


def test_json_set_rejects_unserializable_value():
    with pytest.raises(FieldError):
        JSONSet("data", (["a"], object()))


def test_json_expressions_validate_arguments():
    with pytest.raises(ValueError):
        JSONSet("data")
    with pytest.raises(ValueError):
        JSONRemove("data")
    with pytest.raises(ValueError):
        JSONSet("data", ([], 1))
    with pytest.raises(ValueError):
        JSONSet("data", (["a", -1], 1))
    with pytest.raises(TypeError):
        JSONSet("data", ([True], 1))
    with pytest.raises(TypeError):
        JSONSet("data", ("a.b", 1))


def test_json_update_term_sqlite_sql():
    term = _JSONUpdateTerm(PypikaField("data"), "json_set", [("$.a", "true"), ('$."b.c"', "null")])
    ctx = DEFAULT_SQL_CONTEXT.copy(dialect=Dialects.SQLITE)
    assert term.get_sql(ctx) == """json_set("data", '$.a', json('true'), '$."b.c"', json('null'))"""
    remove = _JSONUpdateTerm(PypikaField("data"), "json_remove", [("$.a", None)])
    assert remove.get_sql(ctx) == 'json_remove("data", \'$.a\')'


def test_json_update_term_parameterizes_path_and_value():
    from pypika_tortoise.terms import Parameterizer

    term = _JSONUpdateTerm(PypikaField("data"), "json_set", [("$.'; DROP--", "true")])
    parameterizer = Parameterizer()
    ctx = DEFAULT_SQL_CONTEXT.copy(dialect=Dialects.SQLITE, parameterizer=parameterizer)
    sql = term.get_sql(ctx)
    assert "DROP" not in sql
    assert parameterizer.values == ["$.'; DROP--", "true"]


def test_json_update_term_unsupported_dialect_raises():
    term = _JSONUpdateTerm(PypikaField("data"), "json_set", [("$.a", "1")])
    for dialect in (Dialects.POSTGRESQL, Dialects.MYSQL, Dialects.MSSQL, Dialects.ORACLE):
        ctx = DEFAULT_SQL_CONTEXT.copy(dialect=dialect)
        with pytest.raises(OperationalError):
            term.get_sql(ctx)
