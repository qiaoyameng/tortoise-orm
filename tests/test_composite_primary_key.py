"""
Tests for explicit composite primary keys (natural keys).

Covers Meta declaration/validation, the public tuple ``pk`` representation,
DDL generation, create/get/save/delete/refresh bound to the full key,
``in_bulk`` over tuples, protection against changing the key of a persisted
instance, rejection of relations pointing at composite-key models, and the
single-column primary-key regression.
"""

import pytest

from tests.testmodels import CompositeTenantItem, SimplePkModel
from tortoise import fields
from tortoise.exceptions import (
    ConfigurationError,
    DoesNotExist,
    FieldError,
    OperationalError,
    ParamsError,
    UnSupportedError,
)
from tortoise.models import Model

pytestmark = pytest.mark.asyncio



async def test_ddl_has_table_level_primary_key(db):
    conn = CompositeTenantItem._meta.db
    rows = await conn.execute_query_dict(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='composite_tenant_item'"
    )
    assert rows
    sql = rows[0]["sql"]
    assert "PRIMARY KEY (\"tenant_id\", \"code\")" in sql
    assert '"tenant_id" INT NOT NULL' in sql


async def test_create_and_get_by_full_key(db):
    await CompositeTenantItem(tenant_id=1, code="X", name="t1").save()
    await CompositeTenantItem(tenant_id=2, code="X", name="t2").save()

    assert await CompositeTenantItem.all().count() == 2

    item = await CompositeTenantItem.get(pk=(1, "X"))
    assert item.name == "t1"
    assert item.pk == (1, "X")

    item_by_fields = await CompositeTenantItem.get(tenant_id=2, code="X")
    assert item_by_fields.name == "t2"

    with pytest.raises(DoesNotExist):
        await CompositeTenantItem.get(pk=(1, "MISSING"))


async def test_getitem_and_missing_key(db):
    await CompositeTenantItem(tenant_id=1, code="X", name="t1").save()
    item = await CompositeTenantItem[(1, "X")]
    assert item.name == "t1"
    with pytest.raises(KeyError):
        await CompositeTenantItem[(99, "X")]


async def test_wrong_pk_shape_raises_field_error(db):
    with pytest.raises(FieldError):
        await CompositeTenantItem.get(pk=("only-one",))
    with pytest.raises(FieldError):
        await CompositeTenantItem.filter(pk__gt=(1, "X"))


async def test_update_is_bound_to_full_key(db):
    first = await CompositeTenantItem.create(tenant_id=1, code="X", name="before")
    second = await CompositeTenantItem.create(tenant_id=2, code="X", name="other")

    first.name = "after"
    await first.save()

    refreshed = await CompositeTenantItem.get(pk=(1, "X"))
    assert refreshed.name == "after"
    assert (await CompositeTenantItem.get(pk=(2, "X"))).name == "other"
    assert second.name == "other"


async def test_update_fields_rejects_pk_components(db):
    item = await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    item.name = "b"
    await item.save(update_fields=["name"])
    assert (await CompositeTenantItem.get(pk=(1, "X"))).name == "b"

    with pytest.raises(OperationalError, match="Can't update pk field"):
        await item.save(update_fields=["code"])


async def test_refresh_and_delete(db):
    item = await CompositeTenantItem.create(tenant_id=5, code="R", name="a")
    await CompositeTenantItem.filter(pk=(5, "R")).update(name="b")
    await item.refresh_from_db()
    assert item.name == "b"

    await item.delete()
    assert not await CompositeTenantItem.filter(pk=(5, "R")).exists()
    with pytest.raises(OperationalError):
        await CompositeTenantItem(tenant_id=9, code="U").delete()


async def test_in_bulk_by_composite_key(db):
    await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    await CompositeTenantItem.create(tenant_id=2, code="X", name="b")
    await CompositeTenantItem.create(tenant_id=1, code="Y", name="c")

    result = await CompositeTenantItem.in_bulk(
        id_list=[(1, "X"), (2, "X"), (3, "missing")]
    )
    assert set(result.keys()) == {(1, "X"), (2, "X")}
    assert result[(2, "X")].name == "b"

    empty = await CompositeTenantItem.in_bulk(id_list=[])
    assert set(empty.keys()) == set()

    whole = await CompositeTenantItem.in_bulk()
    assert set(whole.keys()) == {(1, "X"), (2, "X"), (1, "Y")}


async def test_filter_pk_in(db):
    await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    await CompositeTenantItem.create(tenant_id=2, code="X", name="b")
    rows = await CompositeTenantItem.filter(pk__in=[(1, "X")]).values_list(
        "tenant_id", "code"
    )
    assert rows == [(1, "X")]


async def test_filter_pk_q_object(db):
    from tortoise.expressions import Q

    await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    await CompositeTenantItem.create(tenant_id=2, code="Y", name="b")

    rows = await CompositeTenantItem.filter(
        Q(pk=(1, "X")) | Q(pk=(2, "Y"))
    ).values_list("tenant_id", "code")
    assert sorted(rows) == [(1, "X"), (2, "Y")]

    assert await CompositeTenantItem.filter(
        ~Q(tenant_id=2) & Q(pk=(1, "X"))
    ).exists()


async def test_changing_pk_of_persisted_object_is_rejected(db):
    item = await CompositeTenantItem.create(tenant_id=1, code="X", name="a")

    item.code = "Y"
    with pytest.raises(OperationalError):
        await item.save()
    with pytest.raises(OperationalError):
        await item.save(force_update=True)
    with pytest.raises(OperationalError):
        await item.refresh_from_db()
    with pytest.raises(OperationalError):
        await item.delete()

    assert await CompositeTenantItem.filter(pk=(1, "X")).exists()
    assert not await CompositeTenantItem.filter(pk=(1, "Y")).exists()


async def test_single_pk_modification_is_also_rejected(db):
    obj = await SimplePkModel.create(name="a")
    obj.id = obj.id + 1000
    with pytest.raises(OperationalError):
        await obj.save()
    with pytest.raises(OperationalError):
        await obj.delete()


async def test_clone_composite_pk_requires_tuple(db):
    item = await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    with pytest.raises(ParamsError):
        item.clone()
    clone = item.clone(pk=(2, "Y"))
    await clone.save()
    assert await CompositeTenantItem.filter(pk=(2, "Y")).exists()


async def test_bulk_update_unsupported(db):
    item = await CompositeTenantItem.create(tenant_id=1, code="X", name="a")
    item.name = "b"
    with pytest.raises(UnSupportedError):
        await CompositeTenantItem.bulk_update([item], ["name"])


async def test_single_pk_regression(db):
    obj = await SimplePkModel.create(name="first")
    assert obj.pk == obj.id and obj.pk is not None

    fetched = await SimplePkModel.get(pk=obj.pk)
    assert fetched.name == "first"

    fetched.name = "second"
    await fetched.save()
    assert (await SimplePkModel.get(pk=obj.pk)).name == "second"

    bulk = await SimplePkModel.in_bulk(id_list=[obj.pk, obj.pk + 999])
    assert list(bulk) == [obj.pk]

    await fetched.refresh_from_db()
    await fetched.delete()
    assert await SimplePkModel.all().count() == 0


async def test_relation_targeting_composite_pk_rejected(db_module):
    # Relations are resolved during init, so a fresh context is required.
    import sys
    import types

    from tortoise import Tortoise

    class CpkTarget(Model):
        tid = fields.IntField()
        tcode = fields.CharField(max_length=32)

        class Meta:
            primary_key = ("tid", "tcode")

    class CpkReferrer(Model):
        target = fields.ForeignKeyField("models.CpkTarget")

    module = types.ModuleType("cpk_bad_relations")
    sys.modules["cpk_bad_relations"] = module
    module.CpkTarget = CpkTarget
    module.CpkReferrer = CpkReferrer
    with pytest.raises(ConfigurationError, match="composite primary key"):
        await Tortoise.init(
            db_url="sqlite://:memory:", modules={"models": ["cpk_bad_relations"]}
        )


async def test_m2m_targeting_composite_pk_rejected(db_module):
    import sys
    import types

    from tortoise import Tortoise

    class M2MTarget(Model):
        tid = fields.IntField()
        tcode = fields.CharField(max_length=32)

        class Meta:
            primary_key = ("tid", "tcode")

    class M2MSide(Model):
        targets = fields.ManyToManyField("models.M2MTarget")

    module = types.ModuleType("cpk_bad_m2m")
    sys.modules["cpk_bad_m2m"] = module
    module.M2MTarget = M2MTarget
    module.M2MSide = M2MSide
    with pytest.raises(ConfigurationError, match="composite primary key"):
        await Tortoise.init(
            db_url="sqlite://:memory:", modules={"models": ["cpk_bad_m2m"]}
        )
