from __future__ import annotations

import pytest

from tests.composite_pk import _pipe_selfpipe  # noqa: F401
from tests.testmodels import InheritedCompositeModel


@pytest.mark.asyncio
async def test_composite_pk_inherited_from_abstract(db):
    assert InheritedCompositeModel._meta.composite_pk is True
    assert InheritedCompositeModel._meta.pk_attrs == ("tenant_id", "code")
    await InheritedCompositeModel.create(tenant_id=1, code="a", name="x")
    assert (await InheritedCompositeModel.get(pk=(1, "a"))).name == "x"
