from __future__ import annotations

import pytest

from tests.composite_pk import _pipe_selfpipe  # noqa: F401
from tests.testmodels import CpkFkChild, CpkFkParent


@pytest.mark.asyncio
async def test_composite_model_with_forward_fk(db):
    parent = await CpkFkParent.create(label="root")
    await CpkFkChild.create(tenant_id=1, code="a", parent=parent)
    fetched = await CpkFkChild.get(pk=(1, "a"))
    assert fetched.parent_id == parent.id
    via_filter = await CpkFkChild.filter(parent=parent).all()
    assert {o.pk for o in via_filter} == {(1, "a")}
    children = await parent.cpk_children.all()
    assert len(children) == 1 and children[0].code == "a"
