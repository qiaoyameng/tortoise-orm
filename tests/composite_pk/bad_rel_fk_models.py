"""Models that illegally FK-target a composite primary key."""

from tortoise import fields
from tortoise.models import Model


class BadRelTarget(Model):
    tenant = fields.IntField()
    code = fields.CharField(max_length=10)

    class Meta:
        table = "bad_rel_target_fk"
        primary_key = ("tenant", "code")


class BadRelFkSource(Model):
    target = fields.ForeignKeyField("bad_rel_fk.BadRelTarget")

    class Meta:
        table = "bad_rel_fk_source"
