"""
Models for testing ManyToManyRelation.add() with through_defaults.

The through models are declared explicitly (with extra fields and unique
constraints) and their tables are referenced by name from the M2M fields,
so no schema (re)design is needed: the schema generator skips auto-creating
through tables that already exist as model tables.
"""

from tortoise import fields
from tortoise.models import Model
from tortoise.validators import MinValueValidator


class Member(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50, null=True)

    clubs: fields.ManyToManyRelation["Club"]


class Club(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50, null=True)

    members: fields.ManyToManyRelation[Member] = fields.ManyToManyField(
        "models.Member", related_name="clubs", through="membership"
    )


class Membership(Model):
    id = fields.IntField(primary_key=True)
    club = fields.ForeignKeyField("models.Club", related_name="memberships")
    member = fields.ForeignKeyField("models.Member", related_name="memberships")
    role = fields.CharField(max_length=50, null=True)
    level = fields.IntField(null=True, validators=[MinValueValidator(0)])
    joined_at = fields.DatetimeField(null=True)

    class Meta:
        table = "membership"
        unique_together = (("club", "member"),)


class Node(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50, null=True)

    links: fields.ManyToManyRelation["Node"] = fields.ManyToManyField(
        "models.Node",
        related_name="linked_from",
        through="node_link",
        forward_key="to_node_id",
        backward_key="from_node_id",
    )
    linked_from: fields.ManyToManyRelation["Node"]


class NodeLink(Model):
    id = fields.IntField(primary_key=True)
    from_node = fields.ForeignKeyField("models.Node", related_name="out_links")
    to_node = fields.ForeignKeyField("models.Node", related_name="in_links")
    label = fields.CharField(max_length=50, null=True)

    class Meta:
        table = "node_link"
        unique_together = (("from_node", "to_node"),)


class PlainA(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50, null=True)
    bees: fields.ManyToManyRelation["PlainB"] = fields.ManyToManyField(
        "models.PlainB", related_name="ays"
    )


class PlainB(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50, null=True)

    ays: fields.ManyToManyRelation[PlainA]
