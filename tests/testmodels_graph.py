"""
Models for save_graph tests.

Kept in a dedicated module because the cyclic FK references between
GraphAuthor/GraphBook and GraphLockA/GraphLockB can not be ordered by the
automatic schema generator; tests create these tables per-model instead.
"""

from tortoise import fields
from tortoise.models import Model


class GraphNode(Model):
    name = fields.CharField(max_length=50)


class GraphLink(Model):
    name = fields.CharField(max_length=50)
    left: fields.ForeignKeyRelation[GraphNode] = fields.ForeignKeyField(
        "models.GraphNode", related_name="left_links"
    )
    right: fields.ForeignKeyRelation[GraphNode] = fields.ForeignKeyField(
        "models.GraphNode", related_name="right_links"
    )


class GraphAuthor(Model):
    name = fields.CharField(max_length=50)
    favorite: fields.ForeignKeyNullableRelation["GraphBook"] = fields.ForeignKeyField(
        "models.GraphBook", related_name="favored_by", null=True
    )


class GraphBook(Model):
    title = fields.CharField(max_length=50)
    author: fields.ForeignKeyNullableRelation[GraphAuthor] = fields.ForeignKeyField(
        "models.GraphAuthor", related_name="books", null=True
    )


class GraphLockA(Model):
    b: fields.ForeignKeyRelation["GraphLockB"] = fields.ForeignKeyField(
        "models.GraphLockB", related_name="a_set"
    )


class GraphLockB(Model):
    a: fields.ForeignKeyRelation[GraphLockA] = fields.ForeignKeyField(
        "models.GraphLockA", related_name="b_set"
    )


class GraphUniqueName(Model):
    name = fields.CharField(max_length=50, unique=True)
