from __future__ import annotations

import pytest

from tests.composite_pk import _pipe_selfpipe  # noqa: F401
from tests.testmodels import SourceFieldComposite
from tortoise import Tortoise


@pytest.mark.asyncio
async def test_composite_pk_with_source_fields(db):
    assert SourceFieldComposite._meta.db_pk_columns == ("t_id", "c_code")
    await SourceFieldComposite.create(tenant=1, code="a")
    obj = await SourceFieldComposite.get(pk=(1, "a"))
    assert obj.tenant == 1 and obj.code == "a"
    conn = Tortoise.get_connection("models")
    rows = await conn.execute_query_dict(
        "SELECT t_id, c_code FROM source_field_composite WHERE t_id=? AND c_code=?",
        [1, "a"],
    )
    assert rows == [{"t_id": 1, "c_code": "a"}]
