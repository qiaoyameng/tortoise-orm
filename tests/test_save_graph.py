import os

import pytest
import pytest_asyncio

from tests.testmodels_graph import (
    GraphAuthor,
    GraphBook,
    GraphLink,
    GraphLockA,
    GraphLockB,
    GraphNode,
    GraphUniqueName,
)
from tortoise import save_graph
from tortoise.context import TortoiseContext
from tortoise.exceptions import GraphCycleError, IntegrityError, ParamsError


@pytest_asyncio.fixture(scope="module")
async def graph_db_module():
    """
    Module-scoped context for save_graph tests.

    The graph models contain cyclic FK references, so tables are created
    per-model (SQLite resolves FK targets at DML time, not at CREATE time).
    """
    db_url = os.getenv("TORTOISE_TEST_DB", "sqlite://:memory:")
    ctx = TortoiseContext()
    async with ctx:
        await ctx.init(db_url=db_url, modules={"models": ["tests.testmodels_graph"]})
        conn = ctx.db()
        generator = conn.schema_generator(conn)
        for model in ctx.apps.get_models_iterable():
            table_sql = generator._get_table_sql(model, safe=True)
            await conn.execute_script(table_sql["table_creation_string"])
        yield ctx


@pytest_asyncio.fixture(scope="function")
async def graph_db(graph_db_module):
    """Function-scoped fixture with transaction rollback cleanup."""
    conn = graph_db_module.db()
    transaction = conn._in_transaction()
    await transaction.__aenter__()
    try:
        yield graph_db_module
    finally:

        class _RollbackException(Exception):
            pass

        await transaction.__aexit__(_RollbackException, _RollbackException(), None)


@pytest.mark.asyncio
async def test_diamond_shared_nodes(graph_db):
    root = GraphNode(name="root")
    mid1 = GraphNode(name="mid1")
    mid2 = GraphNode(name="mid2")
    leaf = GraphNode(name="leaf")
    link1 = GraphLink(name="l1")
    link2 = GraphLink(name="l2")
    link3 = GraphLink(name="l3")
    link4 = GraphLink(name="l4")

    objects = [root, mid1, mid2, leaf, link1, link2, link3, link4]
    edges = [
        (link1, "left", mid1),
        (link1, "right", root),
        (link2, "left", mid2),
        (link2, "right", root),
        (link3, "left", leaf),
        (link3, "right", mid1),
        (link4, "left", leaf),
        (link4, "right", mid2),
    ]
    await save_graph(objects, edges)

    for obj in objects:
        assert obj.pk is not None
        assert obj._saved_in_db

    # shared nodes inserted exactly once
    assert await GraphNode.all().count() == 4
    assert await GraphLink.all().count() == 4

    links = {link.name: link async for link in GraphLink.all()}
    assert links["l1"].left_id == mid1.pk
    assert links["l1"].right_id == root.pk
    assert links["l2"].left_id == mid2.pk
    assert links["l2"].right_id == root.pk
    assert links["l3"].left_id == leaf.pk
    assert links["l3"].right_id == mid1.pk
    assert links["l4"].left_id == leaf.pk
    assert links["l4"].right_id == mid2.pk


@pytest.mark.asyncio
async def test_saved_object_is_reference_boundary(graph_db):
    root = await GraphNode.create(name="root")
    child = GraphNode(name="child")
    link = GraphLink(name="l1")

    await save_graph([child, link, root], [(link, "left", child), (link, "right", root)])

    assert await GraphNode.all().count() == 2
    assert await GraphLink.all().count() == 1
    saved_link = await GraphLink.get()
    assert saved_link.left_id == child.pk
    assert saved_link.right_id == root.pk
    # boundary object untouched
    assert (await GraphNode.get(pk=root.pk)).name == "root"


@pytest.mark.asyncio
async def test_nullable_bidirectional_cycle(graph_db):
    author = GraphAuthor(name="alice")
    book = GraphBook(title="tale")

    await save_graph([author, book], [(author, "favorite", book), (book, "author", author)])

    assert author.pk is not None
    assert book.pk is not None
    saved_author = await GraphAuthor.get()
    saved_book = await GraphBook.get()
    assert saved_author.favorite_id == book.pk
    assert saved_book.author_id == author.pk


@pytest.mark.asyncio
async def test_non_nullable_cycle_raises_before_any_write(graph_db):
    lock_a = GraphLockA()
    lock_b = GraphLockB()

    with pytest.raises(GraphCycleError) as excinfo:
        await save_graph([lock_a, lock_b], [(lock_a, "b", lock_b), (lock_b, "a", lock_a)])

    message = str(excinfo.value)
    assert "GraphLockA.b" in message
    assert "GraphLockB.a" in message

    # nothing was written and instances do not look persisted
    assert await GraphLockA.all().count() == 0
    assert await GraphLockB.all().count() == 0
    assert lock_a.pk is None
    assert lock_b.pk is None
    assert not lock_a._saved_in_db
    assert not lock_b._saved_in_db


@pytest.mark.asyncio
async def test_constraint_failure_rolls_back_and_restores_instances(graph_db):
    first = GraphUniqueName(name="dup")
    second = GraphUniqueName(name="dup")

    with pytest.raises(IntegrityError):
        await save_graph([first, second], [])

    # whole graph rolled back
    assert await GraphUniqueName.all().count() == 0
    # in-memory state restored: no leaked pk or saved flag
    assert first.pk is None
    assert second.pk is None
    assert not first._saved_in_db
    assert not second._saved_in_db


@pytest.mark.asyncio
async def test_edge_to_unsaved_object_outside_whitelist_rejected(graph_db):
    root = GraphNode(name="root")
    outsider = GraphNode(name="outsider")
    link = GraphLink(name="l1")

    with pytest.raises(ParamsError):
        await save_graph([root, link], [(link, "left", root), (link, "right", outsider)])

    assert await GraphNode.all().count() == 0
    assert await GraphLink.all().count() == 0
    assert root.pk is None
    assert link.pk is None
