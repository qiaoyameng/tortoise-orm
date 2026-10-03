"""Synchronous validation tests for composite primary key declarations."""

import pytest

from tortoise import fields
from tortoise.exceptions import ConfigurationError
from tortoise.models import Model

from tests.testmodels import CompositeTenantItem


def make_model(name, **meta_kwargs):
    meta = type("Meta", (), meta_kwargs)
    return type(name, (Model,), {"a": fields.IntField(), "Meta": meta})


def test_not_a_sequence():
    with pytest.raises(ConfigurationError):
        make_model("Bad1", primary_key="a")


def test_single_field():
    with pytest.raises(ConfigurationError):
        make_model("Bad2", primary_key=("a",))


def test_duplicate_fields():
    with pytest.raises(ConfigurationError, match="duplicate"):
        make_model("Bad3", primary_key=("a", "a"))


def test_unknown_field():
    with pytest.raises(ConfigurationError, match="unknown field"):
        make_model("Bad4", primary_key=("a", "missing"))


def test_nullable_field():
    with pytest.raises(ConfigurationError, match="can't be nullable"):
        type(
            "Bad5",
            (Model,),
            {
                "a": fields.IntField(),
                "b": fields.IntField(null=True),
                "Meta": type("Meta", (), {"primary_key": ("a", "b")}),
            },
        )


def test_field_also_primary_key():
    with pytest.raises(ConfigurationError, match="part of a composite"):
        type(
            "Bad6",
            (Model,),
            {
                "a": fields.IntField(primary_key=True),
                "b": fields.IntField(),
                "Meta": type("Meta", (), {"primary_key": ("a", "b")}),
            },
        )


def test_generated_field():
    with pytest.raises(ConfigurationError, match="DB-generated"):
        type(
            "Bad7",
            (Model,),
            {
                "a": fields.IntField(generated=True),
                "b": fields.IntField(),
                "Meta": type("Meta", (), {"primary_key": ("a", "b")}),
            },
        )


def test_two_inline_primary_keys_still_rejected():
    with pytest.raises(ConfigurationError, match="two primary keys"):
        type(
            "Bad8",
            (Model,),
            {
                "a": fields.IntField(primary_key=True),
                "b": fields.IntField(primary_key=True),
            },
        )


def test_abstract_model_composite_rejected():
    with pytest.raises(ConfigurationError, match="abstract models"):
        type(
            "AbsBad",
            (Model,),
            {
                "a": fields.IntField(),
                "b": fields.IntField(),
                "Meta": type(
                    "Meta", (), {"abstract": True, "primary_key": ("a", "b")}
                ),
            },
        )


class TestDeclaration:
    def test_meta_attrs(self):
        meta = CompositeTenantItem._meta
        assert meta.is_composite_pk is True
        assert meta.pk_attrs == ("tenant_id", "code")
        assert meta.db_pk_columns == ("tenant_id", "code")
        assert meta.pk is None
        assert meta.db_pk_column == ""

    def test_no_inline_pk_and_no_auto_id(self):
        meta = CompositeTenantItem._meta
        assert "id" not in meta.fields_map
        assert not any(field.pk for field in meta.fields_map.values())

    def test_describe(self):
        desc = CompositeTenantItem.describe()
        assert desc["pk_field"] is None
        assert [field["name"] for field in desc["pk_fields"]] == ["tenant_id", "code"]
        data_field_names = [field["name"] for field in desc["data_fields"]]
        assert "tenant_id" not in data_field_names
        assert "code" not in data_field_names
        assert "name" in data_field_names

    def test_pk_tuple_representation(self):
        item = CompositeTenantItem(tenant_id=7, code="abc", name="n")
        assert item.pk == (7, "abc")
        assert bool(item.pk) is True
        assert hash(item) == hash((7, "abc"))
        assert item == CompositeTenantItem(tenant_id=7, code="abc", name="other")

    def test_unset_pk(self):
        item = CompositeTenantItem(code="abc")
        assert item.pk == (None, "abc")
        assert item._has_pk() is False
        with pytest.raises(TypeError):
            hash(item)

    def test_pk_constructor_validates_shape_and_nulls(self):
        with pytest.raises(ValueError):
            CompositeTenantItem(pk=(1,))
        with pytest.raises(ValueError):
            CompositeTenantItem(pk=(1, None))
        item = CompositeTenantItem(pk=(2, "X"))
        assert item.tenant_id == 2
        assert item.code == "X"

    def test_pk_setter(self):
        item = CompositeTenantItem(tenant_id=1, code="a")
        item.pk = (9, "z")
        assert item.tenant_id == 9 and item.code == "z"
        item.pk = None
        assert item.tenant_id is None and item.code is None
        with pytest.raises(ValueError):
            item.pk = (1,)
