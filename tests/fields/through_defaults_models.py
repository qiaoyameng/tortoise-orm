"""
Models for testing ManyToManyRelation.add() with through_defaults.

The through models are defined explicitly (with unique constraints and extra
fields) and their table names are wired into the M2M fields via ``through=``.
"""

from tortoise import fields
from tortoise.models import Model
from tortoise.validators import MinValueValidator


class Member(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50)


class Tag(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50)


class Club(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50)
    members: fields.ManyToManyRelation["Member"] = fields.ManyToManyField(
        "models.Member",
        related_name="clubs",
        through="club_membership",
        backward_key="club_id",
        forward_key="member_id",
    )
    #: Plain M2M without extra through fields (auto-generated through table)
    tags: fields.ManyToManyRelation["Tag"] = fields.ManyToManyField(
        "models.Tag", related_name="clubs"
    )


class ClubMembership(Model):
    id = fields.IntField(primary_key=True)
    club = fields.ForeignKeyField("models.Club", related_name="memberships")
    member = fields.ForeignKeyField("models.Member", related_name="memberships")
    role = fields.CharField(max_length=50)
    priority = fields.IntField(default=0, validators=[MinValueValidator(0)])
    joined_at = fields.DatetimeField(null=True)

    class Meta:
        table = "club_membership"
        unique_together = (("club", "member"),)


class Node(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50)
    links: fields.ManyToManyRelation["Node"] = fields.ManyToManyField(
        "models.Node",
        related_name="linked_from",
        through="node_link",
        backward_key="from_node_id",
        forward_key="to_node_id",
    )


class NodeLink(Model):
    id = fields.IntField(primary_key=True)
    from_node = fields.ForeignKeyField("models.Node", related_name="out_links")
    to_node = fields.ForeignKeyField("models.Node", related_name="in_links")
    weight = fields.FloatField(default=1.0)

    class Meta:
        table = "node_link"
        unique_together = (("from_node", "to_node"),)
