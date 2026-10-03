"""Composite primary key support: model lifecycle bound to the full key."""

from __future__ import annotations

import pytest

from tests.composite_pk import _pipe_selfpipe  # noqa: F401  (sandbox event-loop shim)
from tests.testmodels import CompositePkModel
from tortoise import Tortoise, fields
from tortoise.exceptions import ConfigurationError, IntegrityError, UnSupportedError
from tortoise.models import CompositePK, Model


class TestCompositeIdentity:
    def test_pk_is_composite_tuple(self, db):
        obj = CompositePkModel(tenant_id=1, code="a", name="x")
        assert obj.pk == (1, "a")
        assert isinstance(obj.pk, CompositePK)
        assert obj.pk.as_dict(("tenant_id", "code")) == {"tenant_id": 1, "code": "a"}

    def test_pk_none_when_any_component_missing(self):
        obj = CompositePkModel(tenant_id=1)
        assert obj.pk is None
        assert not obj._has_pk()

    def test_pk_assignment_distributes_components(self):
        obj = CompositePkModel(tenant_id=1, code="a")
        obj.pk = CompositePK(2, "b")
        assert obj.tenant_id == 2
        assert obj.code == "b"

    def test_pk_assignment_length_mismatch(self):
        obj = CompositePkModel(tenant_id=1, code="a")
        with pytest.raises(ValueError):
            obj.pk = (1,)
        with pytest.raises(ValueError):
            obj.pk = (1, "a", "extra")

    @pytest.mark.asyncio
    async def test_identity_equality_hash_repr(self, db):
        await CompositePkModel.create(tenant_id=1, code="a", name="x")
        a = await CompositePkModel.get(pk=(1, "a"))
        assert a == CompositePkModel.construct(_saved_in_db=True, tenant_id=1, code="a")
        assert hash(a) == hash(CompositePK(1, "a"))
        assert repr(a) == "<CompositePkModel: CompositePK(1, 'a')>"


class TestCompositeDeclarations:
    def _define(self, **meta_attrs):
        attrs = {
            "tenant": fields.IntField(),
            "code": fields.CharField(max_length=10),
            "Meta": type("Meta", (), {"table": "x", **meta_attrs}),
        }
        return type("BadModel", (Model,), attrs)

    def test_duplicate_fields_rejected(self):
        with pytest.raises(ConfigurationError, match="duplicate"):
            self._define(primary_key=("tenant", "tenant"))

    def test_too_few_fields_rejected(self):
        with pytest.raises(ConfigurationError, match="at least two"):
            self._define(primary_key=("tenant",))

    def test_string_rejected(self):
        with pytest.raises(ConfigurationError, match="tuple/list"):
            self._define(primary_key="tenant")

    def test_nullable_component_rejected(self):
        attrs = {
            "tenant": fields.IntField(),
            "code": fields.CharField(max_length=10, null=True),
            "Meta": type("Meta", (), {"table": "x", "primary_key": ("tenant", "code")}),
        }
        with pytest.raises(ConfigurationError, match="nullable"):
            type("NullPkModel", (Model,), attrs)

    def test_component_with_primary_key_true_rejected(self):
        attrs = {
            "tenant": fields.IntField(primary_key=True),
            "code": fields.CharField(max_length=10),
            "Meta": type("Meta", (), {"table": "x", "primary_key": ("tenant", "code")}),
        }
        with pytest.raises(ConfigurationError, match="part of composite primary key"):
            type("DoublePkModel", (Model,), attrs)

    def test_unknown_field_rejected(self):
        with pytest.raises(ConfigurationError, match="is not defined"):
            self._define(primary_key=("tenant", "missing"))


class TestCompositeSchema:
    @pytest.mark.asyncio
    async def test_table_primary_key_constraint(self, db):
        conn = Tortoise.get_connection("models")
        rows = await conn.execute_query_dict(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='composite_pk_model'"
        )
        sql = rows[0]["sql"]
        assert "PRIMARY KEY" in sql
        assert '"tenant_id"' in sql and '"code"' in sql
        # component columns are not individually declared PRIMARY KEY inline
        assert "PRIMARY KEY" not in sql.split("PRIMARY KEY", 1)[0]

    @pytest.mark.asyncio
    async def test_duplicate_full_key_rejected_by_db(self, db):
        await CompositePkModel.create(tenant_id=1, code="dup", name="a")
        with pytest.raises(IntegrityError):
            await CompositePkModel.create(tenant_id=1, code="dup", name="b")


@pytest.mark.asyncio
class TestCompositeLifecycle:
    async def test_two_tenants_same_code(self, db):
        a = await CompositePkModel.create(tenant_id=1, code="dup", name="A")
        b = await CompositePkModel.create(tenant_id=2, code="dup", name="B")
        assert a.pk != b.pk
        assert (await CompositePkModel.get(pk=(1, "dup"))).name == "A"
        assert (await CompositePkModel.get(pk=(2, "dup"))).name == "B"

    async def test_getitem(self, db):
        await CompositePkModel.create(tenant_id=3, code="k", name="C")
        obj = await CompositePkModel[(3, "k")]
        assert obj.name == "C"

    async def test_get_missing_raises(self, db):
        with pytest.raises(Exception):
            await CompositePkModel.get(pk=(999, "missing"))

    async def test_update_binds_full_key(self, db):
        await CompositePkModel.create(tenant_id=1, code="u", name="old")
        await CompositePkModel.create(tenant_id=2, code="u", name="other")
        obj = await CompositePkModel.get(pk=(1, "u"))
        obj.name = "new"
        await obj.save()
        assert (await CompositePkModel.get(pk=(1, "u"))).name == "new"
        assert (await CompositePkModel.get(pk=(2, "u"))).name == "other"

    async def test_update_fields(self, db):
        await CompositePkModel.create(tenant_id=1, code="f", name="n1")
        obj = await CompositePkModel.get(pk=(1, "f"))
        obj.name = "n2"
        await obj.save(update_fields=["name"])
        assert (await CompositePkModel.get(pk=(1, "f"))).name == "n2"

    async def test_update_fields_cannot_include_pk(self, db):
        await CompositePkModel.create(tenant_id=1, code="p", name="n")
        obj = await CompositePkModel.get(pk=(1, "p"))
        with pytest.raises(Exception):
            await obj.save(update_fields=["tenant_id"])

    async def test_pk_mutation_on_saved_instance_rejected(self, db):
        await CompositePkModel.create(tenant_id=1, code="m", name="a")
        await CompositePkModel.create(tenant_id=2, code="m", name="b")
        obj = await CompositePkModel.get(pk=(1, "m"))
        obj.tenant_id = 2
        with pytest.raises(IntegrityError, match="modified primary key"):
            await obj.save()
        with pytest.raises(IntegrityError):
            await obj.save(force_update=True)
        # untouched row still exists
        assert (await CompositePkModel.get(pk=(2, "m"))).name == "b"

    async def test_refresh_from_db(self, db):
        await CompositePkModel.create(tenant_id=1, code="r", name="first")
        obj = await CompositePkModel.get(pk=(1, "r"))
        await CompositePkModel.filter(pk=(1, "r")).update(name="second")
        await obj.refresh_from_db()
        assert obj.name == "second"

    async def test_delete_removes_only_matching_row(self, db):
        await CompositePkModel.create(tenant_id=1, code="d", name="a")
        await CompositePkModel.create(tenant_id=2, code="d", name="b")
        await (await CompositePkModel.get(pk=(1, "d"))).delete()
        remaining = await CompositePkModel.all()
        assert len(remaining) == 1
        assert remaining[0].tenant_id == 2

    async def test_in_bulk(self, db):
        await CompositePkModel.create(tenant_id=1, code="i1", name="one")
        await CompositePkModel.create(tenant_id=2, code="i2", name="two")
        result = await CompositePkModel.in_bulk([(1, "i1"), (2, "i2"), (3, "nope")])
        assert set(result) == {CompositePK(1, "i1"), CompositePK(2, "i2")}
        assert result[(1, "i1")].name == "one"

    async def test_in_bulk_wrong_part_width(self, db):
        with pytest.raises(Exception):
            await CompositePkModel.in_bulk([(1,)])

    async def test_pk_in_filter(self, db):
        await CompositePkModel.create(tenant_id=1, code="a", name="a")
        await CompositePkModel.create(tenant_id=1, code="b", name="b")
        await CompositePkModel.create(tenant_id=2, code="a", name="c")
        rows = await CompositePkModel.filter(pk__in=[(1, "a"), (2, "a")]).all()
        assert {o.name for o in rows} == {"a", "c"}

    async def test_filter_by_model_instance(self, db):
        await CompositePkModel.create(tenant_id=1, code="z", name="z")
        detached = CompositePkModel(tenant_id=1, code="z")
        row = await CompositePkModel.filter(pk=detached).first()
        assert row is not None and row.name == "z"

    async def test_contains(self, db):
        await CompositePkModel.create(tenant_id=1, code="c", name="x")
        obj = await CompositePkModel.get(pk=(1, "c"))
        assert await CompositePkModel.all().contains(obj)
        assert not await CompositePkModel.filter(tenant_id=2).contains(obj)

    async def test_clone_requires_explicit_key(self, db):
        await CompositePkModel.create(tenant_id=1, code="o", name="orig")
        obj = await CompositePkModel.get(pk=(1, "o"))
        with pytest.raises(Exception):
            obj.clone()
        clone = obj.clone(pk=(5, "n"))
        await clone.save()
        assert await CompositePkModel.get(pk=(5, "n"))

    async def test_bulk_create_composite(self, db):
        await CompositePkModel.bulk_create(
            [
                CompositePkModel(tenant_id=10, code="b1", name="one"),
                CompositePkModel(tenant_id=10, code="b2", name="two"),
                CompositePkModel(tenant_id=11, code="b1", name="three"),
            ]
        )
        assert await CompositePkModel.filter(tenant_id__in=[10, 11]).count() == 3

    async def test_bulk_update_unsupported(self, db):
        await CompositePkModel.create(tenant_id=1, code="b", name="x")
        obj = await CompositePkModel.construct(_saved_in_db=True, tenant_id=1, code="b", name="y")
        with pytest.raises(UnSupportedError):
            await CompositePkModel.bulk_update([obj], fields=["name"])


class TestRelationToComposite:
    @pytest.mark.asyncio
    async def test_fk_to_composite_rejected_at_init(self):
        with pytest.raises(ConfigurationError, match="composite primary key"):
            await Tortoise.init(
                db_url="sqlite://:memory:",
                modules={"bad_rel_fk": ["tests.composite_pk.bad_rel_fk_models"]},
            )

    @pytest.mark.asyncio
    async def test_m2m_on_composite_model_rejected_at_init(self):
        with pytest.raises(ConfigurationError, match="composite primary key"):
            await Tortoise.init(
                db_url="sqlite://:memory:",
                modules={"bad_rel_m2m": ["tests.composite_pk.bad_rel_m2m_models"]},
            )
