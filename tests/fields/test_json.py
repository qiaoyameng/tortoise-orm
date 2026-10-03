from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pypika_tortoise import Table

from tests import testmodels
from tortoise.backends.base.client import Capabilities
from tortoise.contrib.test import requireCapability
from tortoise.contrib.test.condition import In
from tortoise.exceptions import (
    ConfigurationError,
    DoesNotExist,
    FieldError,
    IntegrityError,
    UnSupportedError,
)
from tortoise.expressions import JSONRemove, JSONSet, ResolveContext
from tortoise.fields import JSONField
from tortoise.indexes import Index


@pytest.mark.asyncio
async def test_empty(db):
    """Test that creating without required JSON field raises IntegrityError."""
    with pytest.raises(IntegrityError):
        await testmodels.JSONFields.create()


class MyPydanticModel(BaseModel):
    name: str
    idx: Index
    model_config = dict(arbitrary_types_allowed=True)


@pytest.mark.asyncio
async def test_create(db):
    """Test JSON field creation and retrieval."""
    obj0 = await testmodels.JSONFields.create(data={"some": ["text", 3]})
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data == {"some": ["text", 3]}
    assert obj.data_null is None
    await obj.save()
    obj2 = await testmodels.JSONFields.get(id=obj.id)
    assert obj == obj2
    obj3 = await testmodels.JSONFields.create(data="{}", data_decimal=Decimal(0))
    obj3 = await testmodels.JSONFields.get(id=obj3.id)
    assert str(obj3.data_decimal) == "0"
    pyd_model = MyPydanticModel(name="", idx=Index(fields=["data"]))
    obj4 = await testmodels.JSONFields.create(data="{}", data_index=pyd_model)
    obj4 = await testmodels.JSONFields.get(id=obj4.id)
    assert obj4.data_index == {
        "idx": {"expressions": [], "extra": "", "fields": ["data"], "name": None, "type": ""},
        "name": "",
    }


@pytest.mark.asyncio
async def test_error(db):
    """Test that invalid JSON raises FieldError."""
    with pytest.raises(FieldError):
        await testmodels.JSONFields.create(data='{"some": ')

    obj = await testmodels.JSONFields.create(data='{"some": ["text", 3]}')
    with pytest.raises(FieldError):
        await testmodels.JSONFields.filter(pk=obj.pk).update(data='{"some": ')

    with pytest.raises(FieldError):
        obj.data = "error json"
        await obj.save()

    with pytest.raises(TypeError):
        await testmodels.JSONFields.create(data=Decimal(0))

    with pytest.raises(TypeError):
        await testmodels.JSONFields.create(data_decimal=Index(fields=["data"]))


@pytest.mark.asyncio
async def test_update(db):
    """Test JSON field update."""
    obj0 = await testmodels.JSONFields.create(data={"some": ["text", 3]})
    await testmodels.JSONFields.filter(id=obj0.id).update(data={"other": ["text", 5]})
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data == {"other": ["text", 5]}
    assert obj.data_null is None


@pytest.mark.asyncio
async def test_dict_str(db):
    """Test JSON field with dict from string."""
    obj0 = await testmodels.JSONFields.create(data={"some": ["text", 3]})

    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data == {"some": ["text", 3]}

    await testmodels.JSONFields.filter(id=obj0.id).update(data='{"other": ["text", 5]}')
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data == {"other": ["text", 5]}


@pytest.mark.asyncio
async def test_list_str(db):
    """Test JSON field with list from string."""
    obj = await testmodels.JSONFields.create(data='["text", 3]')
    obj0 = await testmodels.JSONFields.get(id=obj.id)
    assert obj0.data == ["text", 3]

    await testmodels.JSONFields.filter(id=obj.id).update(data='["text", 5]')
    obj0 = await testmodels.JSONFields.get(id=obj.id)
    assert obj0.data == ["text", 5]


@pytest.mark.asyncio
async def test_list(db):
    """Test JSON field with list data."""
    obj0 = await testmodels.JSONFields.create(data=["text", 3])
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data == ["text", 3]
    assert obj.data_null is None
    await obj.save()
    obj2 = await testmodels.JSONFields.get(id=obj.id)
    assert obj == obj2


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_list_contains(db):
    """Test JSON contains filter on list."""
    await testmodels.JSONFields.create(data=["text", 3, {"msg": "msg2"}])
    obj = await testmodels.JSONFields.filter(data__contains=[{"msg": "msg2"}]).first()
    assert obj.data == ["text", 3, {"msg": "msg2"}]
    await obj.save()
    obj2 = await testmodels.JSONFields.get(id=obj.id)
    assert obj == obj2


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_list_contained_by(db):
    """Test JSON contained_by filter on list."""
    obj0 = await testmodels.JSONFields.create(data=["text"])
    obj1 = await testmodels.JSONFields.create(data=["tortoise", "msg"])
    obj2 = await testmodels.JSONFields.create(data=["tortoise"])
    obj3 = await testmodels.JSONFields.create(data=["new_message", "some_message"])
    objs = set(await testmodels.JSONFields.filter(data__contained_by=["text", "tortoise", "msg"]))
    created_objs = {obj0, obj1, obj2}
    assert created_objs == objs
    assert obj3 not in objs


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_filter(db):
    """Test JSON filter with nested data."""
    obj0 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": "Bob",
                "last": None,
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )
    obj1 = await testmodels.JSONFields.create(
        data={
            "breed": "husky",
            "owner": {
                "name": "Goldast",
                "last": None,
                "other_pets": [
                    {
                        "name": None,
                    }
                ],
            },
        }
    )
    obj = await testmodels.JSONFields.get(data__filter={"breed": "labrador"})
    obj2 = await testmodels.JSONFields.get(data__filter={"owner__name": "Goldast"})
    obj3 = await testmodels.JSONFields.get(data__filter={"owner__other_pets__0__name": "Fishy"})

    assert obj0 == obj
    assert obj1 == obj2
    assert obj0 == obj3

    with pytest.raises(DoesNotExist):
        await testmodels.JSONFields.get(data__filter={"breed": "NotFound"})
    with pytest.raises(DoesNotExist):
        await testmodels.JSONFields.get(data__filter={"owner__other_pets__0__name": "NotFound"})


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_filter_not_condition(db):
    """Test JSON filter with not condition."""
    obj0 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": "Bob",
                "last": None,
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )
    obj1 = await testmodels.JSONFields.create(
        data={
            "breed": "husky",
            "owner": {
                "name": "Goldast",
                "last": None,
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )

    obj2 = await testmodels.JSONFields.get(data__filter={"breed__not": "husky"})
    obj3 = await testmodels.JSONFields.get(data__filter={"breed__not": "labrador"})
    assert obj0 == obj2
    assert obj1 == obj3


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_filter_is_null_condition(db):
    """Test JSON filter with isnull condition."""
    obj0 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": "Boby",
                "last": "Cloud",
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )

    obj1 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": None,
                "last": "Cloud",
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )

    obj2 = await testmodels.JSONFields.get(data__filter={"owner__name__isnull": False})
    obj3 = await testmodels.JSONFields.get(data__filter={"owner__name__isnull": True})
    assert obj0 == obj2
    assert obj1 == obj3


@requireCapability(dialect=In("mysql", "postgres"))
@pytest.mark.asyncio
async def test_filter_not_is_null_condition(db):
    """Test JSON filter with not_isnull condition."""
    obj0 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": "Boby",
                "last": "Cloud",
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )

    obj1 = await testmodels.JSONFields.create(
        data={
            "breed": "labrador",
            "owner": {
                "name": None,
                "last": "Cloud",
                "other_pets": [
                    {
                        "name": "Fishy",
                    }
                ],
            },
        }
    )

    obj2 = await testmodels.JSONFields.get(data__filter={"owner__name__not_isnull": True})
    obj3 = await testmodels.JSONFields.get(data__filter={"owner__name__not_isnull": False})
    assert obj0 == obj2
    assert obj1 == obj3


@pytest.mark.asyncio
async def test_values(db):
    """Test JSON field in values()."""
    obj0 = await testmodels.JSONFields.create(data={"some": ["text", 3]})
    values = await testmodels.JSONFields.filter(id=obj0.id).values("data")
    assert values[0]["data"] == {"some": ["text", 3]}


@pytest.mark.asyncio
async def test_values_list(db):
    """Test JSON field in values_list()."""
    obj0 = await testmodels.JSONFields.create(data={"some": ["text", 3]})
    values = await testmodels.JSONFields.filter(id=obj0.id).values_list("data", flat=True)
    assert values[0] == {"some": ["text", 3]}


def test_unique_fail():
    """Test that JSONField cannot be unique."""
    with pytest.raises(ConfigurationError, match="can't be indexed"):
        JSONField(unique=True)


def test_index_fail():
    """Test that JSONField cannot be indexed."""
    with pytest.raises(ConfigurationError, match="can't be indexed"):
        with pytest.warns(
            DeprecationWarning, match="`index` is deprecated, please use `db_index` instead"
        ):
            JSONField(index=True)
    with pytest.raises(ConfigurationError, match="can't be indexed"):
        JSONField(db_index=True)


@pytest.mark.asyncio
async def test_validate_str(db):
    """Test JSON field with validate from string."""
    obj0 = await testmodels.JSONFields.create(data=[], data_validate='["text", 5]')
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data_validate == ["text", 5]


@pytest.mark.asyncio
async def test_validate_dict(db):
    """Test JSON field with validate from dict."""
    obj0 = await testmodels.JSONFields.create(data=[], data_validate={"some": ["text", 3]})
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data_validate == {"some": ["text", 3]}


@pytest.mark.asyncio
async def test_validate_list(db):
    """Test JSON field with validate from list."""
    obj0 = await testmodels.JSONFields.create(data=[], data_validate=["text", 3])
    obj = await testmodels.JSONFields.get(id=obj0.id)
    assert obj.data_validate == ["text", 3]


@pytest.mark.asyncio
async def test_json_set_queryset_nested_paths_and_types(db):
    obj = await testmodels.JSONFields.create(data={"profile": {"name": "Ada"}, "items": [1]})

    await testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ("profile", "a.b", 'quote."'), True)
    )
    await testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ("items", 1), {"ok": False})
    )
    await testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ("values",), None)
    )

    refreshed = await testmodels.JSONFields.get(pk=obj.pk)
    assert refreshed.data == {
        "profile": {"name": "Ada", "a": {"b": {"quote.\"": True}}},
        "items": [1, {"ok": False}],
        "values": None,
    }


@pytest.mark.asyncio
async def test_json_set_save_nested_paths_and_types(db):
    obj = await testmodels.JSONFields.create(data={"profile": {}})
    obj.data = JSONSet("data", ("profile", "score"), 1.5)
    await obj.save(update_fields=["data"])

    refreshed = await testmodels.JSONFields.get(pk=obj.pk)
    assert refreshed.data == {"profile": {"score": 1.5}}


@pytest.mark.asyncio
async def test_json_two_updates_keep_both_changes(db):
    obj = await testmodels.JSONFields.create(data={"shared": {"first": 0, "second": 0}})

    await testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ("shared", "first"), 1)
    )
    await testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", ("shared", "second"), 2)
    )

    refreshed = await testmodels.JSONFields.get(pk=obj.pk)
    assert refreshed.data == {"shared": {"first": 1, "second": 2}}


@pytest.mark.asyncio
async def test_json_set_remove_ordered_multiple_operations(db):
    obj = await testmodels.JSONFields.create(data={"a": 1, "b": 2, "items": [10, 20]})
    expression = JSONRemove(
        JSONSet(
            "data",
            [
                (("c",), 3),
                (("d",), 4),
                (("c",), 5),
                (("e",), 6),
            ],
        ),
        ("a",),
        ("b",),
        ("e",),
        ("items",),
    )

    await testmodels.JSONFields.filter(pk=obj.pk).update(data=expression)

    refreshed = await testmodels.JSONFields.get(pk=obj.pk)
    assert refreshed.data == {"c": 5, "d": 4}


@pytest.mark.asyncio
async def test_json_set_and_remove_boundary_behavior(db):
    await testmodels.JSONFields.create(id=1, data={"parent": [1]})
    await testmodels.JSONFields.create(id=2, data={})
    await testmodels.JSONFields.create(id=3, data_null=None)

    await testmodels.JSONFields.filter(pk=1).update(
        data=JSONSet("data", ("parent", 3), "ignored")
    )
    await testmodels.JSONFields.filter(pk=2).update(
        data=JSONSet("data", ("missing", "child", 0), "created")
    )
    await testmodels.JSONFields.filter(pk=2).update(
        data=JSONRemove("data", ("missing", "other"))
    )
    await testmodels.JSONFields.filter(pk=3).update(
        data=JSONSet("data_null", ("a",), True)
    )
    await testmodels.JSONFields.filter(pk=3).update(
        data=JSONRemove("data_null", ("a",))
    )

    assert (await testmodels.JSONFields.get(pk=1)).data == {"parent": [1]}
    assert (await testmodels.JSONFields.get(pk=2)).data == {
        "missing": {"child": ["created"]}
    }
    assert (await testmodels.JSONFields.get(pk=3)).data_null is None


@pytest.mark.asyncio
async def test_json_structured_paths_are_bound_parameters(db):
    malicious_key = "a'), (SELECT 1); --"
    obj = await testmodels.JSONFields.create(data={})
    queryset = testmodels.JSONFields.filter(pk=obj.pk).update(
        data=JSONSet("data", (malicious_key,), "safe")
    )

    sql = queryset.sql()
    assert malicious_key not in sql
    assert sql.count("?") == 4
    await queryset

    refreshed = await testmodels.JSONFields.get(pk=obj.pk)
    assert refreshed.data == {malicious_key: "safe"}


def test_json_set_rejects_invalid_path():
    with pytest.raises(FieldError, match="sequences"):
        JSONSet("data", "not.a.path", 1)
    with pytest.raises(FieldError, match="non-negative"):
        JSONSet("data", (-1,), 1)
    with pytest.raises(FieldError, match="at least one"):
        JSONRemove("data", ())


def test_json_set_reports_unsupported_backend():
    field = JSONField()
    model = type(
        "JSONModel",
        (),
        {
            "__name__": "JSONModel",
            "_meta": SimpleNamespace(
                fields_map={"data": field},
                fields_db_projection={"data": "data"},
                db=SimpleNamespace(
                    capabilities=Capabilities(
                        "postgres",
                        support_json_path_update=False,
                    )
                ),
            ),
        },
    )

    with pytest.raises(UnSupportedError, match="postgres does not support JSON path"):
        JSONSet("data", ("a",), 1).resolve(
            ResolveContext(
                model=model,
                table=Table("json_model"),
                annotations={},
                custom_filters={},
            )
        )
