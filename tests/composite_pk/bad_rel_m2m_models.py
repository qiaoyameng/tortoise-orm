"""A model with a composite primary key that illegally declares ManyToMany."""

from tortoise import fields
from tortoise.models import Model


class BadRelM2mOther(Model):
    id = fields.IntField(primary_key=True)

    class Meta:
        table = "bad_rel_m2m_other"


class BadRelM2mComposite(Model):
    tenant = fields.IntField()
    code = fields.CharField(max_length=10)
    targets = fields.ManyToManyField("bad_rel_m2m.BadRelM2mOther")

    class Meta:
        table = "bad_rel_m2m_composite"
        primary_key = ("tenant", "code")
