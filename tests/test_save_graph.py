"""Tests for transactional graph persistence: ``tortoise.save_graph``."""

import pytest

from tortoise import Edge, save_graph
from tortoise.exceptions import ConfigurationError, IntegrityError

from tests.testmodels import (
    GraphLeft,
    GraphNullableA,
    GraphRequiredLink,
    GraphRight,
    GraphShared,
    GraphTop,
    GraphUniqueName,
)

def _with_relation(instance, field, target):
    """Pre-assign an (unsaved) relation object bypassing __init__ checks."""
    fk = type(instance)._meta.fields_map[field]
    object.__setattr__(instance, f"_{field}", target)
    object.__setattr__(instance, fk.source_field, target.pk if target else None)
    return instance



@pytest.mark.asyncio
async def test_save_graph_diamond_shared_node_inserted_once(db):
    shared = GraphShared(name="shared")
    left = GraphLeft(name="left")
    right = GraphRight(name="right")
    top = GraphTop(name="top")

    inserted = await save_graph(
        [top, left, right, shared],
        [
            Edge(left, "shared", shared),
            Edge(right, "shared", shared),
            Edge(top, "left", left),
            Edge(top, "right", right),
        ],
    )

    # All five... four objects inserted exactly once.
    assert len(inserted) == 4
    assert len({id(o) for o in inserted}) == 4
    for obj in inserted:
        assert obj._saved_in_db
        assert obj.pk is not None

    assert await GraphShared.all().count() == 1
    assert await GraphLeft.all().count() == 1
    assert await GraphRight.all().count() == 1
    assert await GraphTop.all().count() == 1

    # Both branches point at the same shared row.
    left_db = await GraphLeft.get(name="left")
    right_db = await GraphRight.get(name="right")
    top_db = await GraphTop.get(name="top")
    assert left_db.shared_id == shared.pk
    assert right_db.shared_id == shared.pk
    assert top_db.left_id == left.pk
    assert top_db.right_id == right.pk


@pytest.mark.asyncio
async def test_save_graph_edges_to_unsaved_without_prefetch(db):
    """Edges work even though the relation assignment API rejects unsaved."""
    parent = GraphShared(name="parent")
    child = GraphLeft(name="child")

    # The normal assignment path refuses unsaved instances; save_graph must not
    # depend on it.
    with pytest.raises(Exception):
        GraphLeft(name="x", shared=parent)

    await save_graph([child, parent], [Edge(child, "shared", parent)])

    child_db = await GraphLeft.get(name="child")
    assert child_db.shared_id == parent.pk
    assert child._saved_in_db and parent._saved_in_db


@pytest.mark.asyncio
async def test_save_graph_saved_instance_is_reference_boundary(db):
    shared = await GraphShared.create(name="already-there")
    left = GraphLeft(name="child-of-existing")

    inserted = await save_graph([left], [Edge(left, "shared", shared)])
    assert inserted == [left]
    assert await GraphShared.all().count() == 1
    assert (await GraphLeft.get(name="child-of-existing")).shared_id == shared.pk


@pytest.mark.asyncio
async def test_save_graph_traversal_whitelist(db):
    shared = GraphShared(name="s")
    left = _with_relation(GraphLeft(name="l"), "shared", shared)
    right = _with_relation(GraphRight(name="r"), "shared", shared)

    # Without a whitelist the shared objects are not saved implicitly:
    with pytest.raises(IntegrityError):
        await save_graph([GraphLeft(name="orphan-l")])

    # Whitelist via model classes follows declared relations.
    await save_graph([left, right], traverse=[GraphLeft, GraphRight])
    assert await GraphShared.all().count() == 1
    assert (await GraphLeft.get(name="l")).shared_id == shared.pk
    assert (await GraphRight.get(name="r")).shared_id == shared.pk


@pytest.mark.asyncio
async def test_save_graph_traversal_string_and_tuple_whitelist(db):
    shared = GraphShared(name="s2")
    left = _with_relation(GraphLeft(name="l2"), "shared", shared)

    await save_graph([left], traverse=["GraphLeft.shared"])

    fresh = GraphShared(name="s3")
    right = _with_relation(GraphRight(name="r3"), "shared", fresh)
    await save_graph([right], traverse=[(GraphRight, "shared")])

    assert await GraphShared.filter(name__in=["s2", "s3"]).count() == 2


@pytest.mark.asyncio
async def test_save_graph_traversal_multi_level_chain(db):
    shared = GraphShared(name="deep-s")
    left = _with_relation(GraphLeft(name="deep-l"), "shared", shared)
    right = _with_relation(GraphRight(name="deep-r"), "shared", shared)
    top = _with_relation(GraphTop(name="deep-t"), "left", left)
    top = _with_relation(top, "right", right)

    # Only the top is a root; traversal follows top.left -> left.shared and
    # top.right -> right.shared (one shared diamond node, reached twice).
    await save_graph([top], traverse=[GraphTop, GraphLeft, GraphRight])
    assert await GraphShared.all().count() == 1
    top_db = await GraphTop.get(name="deep-t")
    assert top_db.left_id == left.pk
    assert top_db.right_id == right.pk
    assert (await GraphLeft.get(name="deep-l")).shared_id == shared.pk


@pytest.mark.asyncio
async def test_save_graph_traversal_app_qualified_string(db):
    shared = GraphShared(name="appq")
    left = _with_relation(GraphLeft(name="appq-l"), "shared", shared)
    await save_graph([left], traverse=["models.GraphLeft.shared"])
    assert (await GraphLeft.get(name="appq-l")).shared_id == shared.pk


@pytest.mark.asyncio
async def test_save_graph_whitelist_does_not_save_unrelated(db):
    # shared is reachable via left; right is a separate unsaved object that is
    # never referenced and must therefore not be inserted.
    shared = GraphShared(name="reachable")
    left = _with_relation(GraphLeft(name="has-shared"), "shared", shared)
    unrelated = GraphRight(name="never-referenced")

    await save_graph([left], traverse=[GraphLeft])
    assert await GraphRight.filter(name="never-referenced").count() == 0
    assert unrelated.pk is None
    assert unrelated._saved_in_db is False


@pytest.mark.asyncio
async def test_save_graph_nullable_bidirectional_cycle(db):
    a = GraphNullableA(name="a")
    b = GraphNullableA(name="b")

    inserted = await save_graph([a, b], [Edge(a, "peer", b), Edge(b, "peer", a)])
    assert len(inserted) == 2

    a_db = await GraphNullableA.get(name="a")
    b_db = await GraphNullableA.get(name="b")
    assert a_db.peer_id == b.pk
    assert b_db.peer_id == a.pk

    # In-memory instances reflect the final state.
    assert a.peer is b
    assert b.peer is a


@pytest.mark.asyncio
async def test_save_graph_nullable_cycle_with_extra_back_edge(db):
    # Cycle a -> b -> a plus an additional nullable edge a -> c -> a so that
    # breaking the cycle may require deferring more than one edge on one node.
    a = GraphNullableA(name="ma")
    b = GraphNullableA(name="mb")
    c = GraphNullableA(name="mc")
    await save_graph(
        [a, b, c],
        [
            Edge(a, "peer", b),
            Edge(b, "peer", a),
            Edge(c, "peer", a),
        ],
    )
    rows = {
        r["name"]: r["peer_id"]
        for r in await GraphNullableA.filter(name__in=["ma", "mb", "mc"]).values(
            "name", "peer_id"
        )
    }
    assert rows == {"ma": b.pk, "mb": a.pk, "mc": a.pk}


@pytest.mark.asyncio
async def test_save_graph_nullable_self_cycle(db):
    only = GraphNullableA(name="self-loop")
    await save_graph([only], [Edge(only, "peer", only)])
    assert (await GraphNullableA.get(name="self-loop")).peer_id == only.pk


@pytest.mark.asyncio
async def test_save_graph_null_edge_clears_relation(db):
    a = GraphNullableA(name="lonely")
    await save_graph([a], [Edge(a, "peer", None)])
    assert (await GraphNullableA.get(name="lonely")).peer_id is None


@pytest.mark.asyncio
async def test_save_graph_illegal_nonnullable_cycle_diagnosed_before_write(db):
    x = GraphRequiredLink(name="x")
    y = GraphRequiredLink(name="y")
    z = GraphRequiredLink(name="z")

    with pytest.raises(ConfigurationError) as exc:
        await save_graph(
            [x, y, z],
            [Edge(x, "other", y), Edge(y, "other", z), Edge(z, "other", x)],
        )

    message = str(exc.value)
    assert "cycle" in message.lower()
    assert "GraphRequiredLink.other" in message
    # Nothing was written.
    assert await GraphRequiredLink.all().count() == 0
    for obj in (x, y, z):
        assert obj.pk is None
        assert obj._saved_in_db is False


@pytest.mark.asyncio
async def test_save_graph_nonnullable_self_cycle(db):
    node = GraphRequiredLink(name="me")
    with pytest.raises(ConfigurationError):
        await save_graph([node], [Edge(node, "other", node)])
    assert await GraphRequiredLink.all().count() == 0
    assert node.pk is None


@pytest.mark.asyncio
async def test_save_graph_constraint_failure_restores_db_and_instances(db):
    good_shared = GraphShared(name="good")
    # Second unique-name row collides with an already persisted one.
    await GraphUniqueName.create(name="dup", required=await GraphShared.create(name="seed"))

    colliding = GraphUniqueName(name="dup")
    # Make the FK valid at insert time, but the UNIQUE(name) constraint fails.
    inserted_order = []

    original_save = GraphUniqueName.save

    async def failing_save(self, *args, **kwargs):  # noqa: ANN001
        inserted_order.append(self)
        return await original_save(self, *args, **kwargs)

    GraphUniqueName.save = failing_save
    try:
        with pytest.raises(IntegrityError):
            await save_graph(
                [colliding, good_shared],
                [Edge(colliding, "required", good_shared)],
            )
    finally:
        GraphUniqueName.save = original_save

    # DB rolled back: the shared row inserted earlier in the same transaction
    # must be gone too.
    assert await GraphShared.filter(name="good").count() == 0
    assert await GraphUniqueName.filter(name="dup").count() == 1

    # Instances must not look persisted.
    assert colliding.pk is None
    assert colliding._saved_in_db is False
    assert good_shared.pk is None
    assert good_shared._saved_in_db is False


@pytest.mark.asyncio
async def test_save_graph_cycle_failure_restores_deferred_state(db):
    a = GraphNullableA(name="cyc-a")
    b = GraphNullableA(name="cyc-b")

    original_execute_update = type(a)._meta.db.executor_class.execute_update

    async def boom(self, instance, update_fields):  # noqa: ANN001
        if (
            isinstance(instance, GraphNullableA)
            and update_fields == ("peer_id",)
        ):
            raise IntegrityError("simulated backfill failure")
        return await original_execute_update(self, instance, update_fields)

    type(a)._meta.db.executor_class.execute_update = boom
    try:
        with pytest.raises(IntegrityError):
            await save_graph([a, b], [Edge(a, "peer", b), Edge(b, "peer", a)])
    finally:
        type(a)._meta.db.executor_class.execute_update = original_execute_update

    assert await GraphNullableA.all().count() == 0
    for obj in (a, b):
        assert obj.pk is None
        assert obj._saved_in_db is False
        assert not hasattr(obj, "_peer") or getattr(obj, "_peer", None) is None


@pytest.mark.asyncio
async def test_save_graph_edge_from_saved_instance_rejected(db):
    saved = await GraphNullableA.create(name="persisted")
    fresh = GraphNullableA(name="fresh")
    with pytest.raises(ConfigurationError):
        await save_graph([saved, fresh], [Edge(saved, "peer", fresh)])
    assert fresh.pk is None


@pytest.mark.asyncio
async def test_save_graph_empty_objects(db):
    assert await save_graph([]) == []


@pytest.mark.asyncio
async def test_save_graph_invalid_edge_field(db):
    node = GraphNullableA(name="n")
    with pytest.raises(ConfigurationError):
        await save_graph([node], [Edge(node, "name", None)])
    assert node.pk is None
    assert await GraphNullableA.all().count() == 0


@pytest.mark.asyncio
async def test_save_graph_nonnullable_cycle_path_lists_all_nodes(db):
    x = GraphRequiredLink(name="x")
    y = GraphRequiredLink(name="y")
    z = GraphRequiredLink(name="z")
    with pytest.raises(ConfigurationError) as exc:
        await save_graph(
            [x, y, z],
            [Edge(x, "other", y), Edge(y, "other", z), Edge(z, "other", x)],
        )
    message = str(exc.value)
    # The full path x -> y -> z -> x is reported before any write.
    assert message.count("GraphRequiredLink.other") == 3
    assert "back to GraphRequiredLink" in message


@pytest.mark.asyncio
async def test_save_graph_edge_target_must_match_relation(db):
    a = GraphNullableA(name="a")
    other_model = GraphShared(name="wrong-type")
    with pytest.raises(Exception):
        await save_graph([a, other_model], [Edge(a, "peer", other_model)])
    assert await GraphNullableA.all().count() == 0


@pytest.mark.asyncio
async def test_save_graph_explicit_edge_source_must_be_in_objects(db):
    a = GraphNullableA(name="a")
    b = GraphNullableA(name="b")
    with pytest.raises(ConfigurationError):
        await save_graph([a], [Edge(b, "peer", a)])
    assert a.pk is None and b.pk is None


@pytest.mark.asyncio
async def test_save_graph_no_implicit_save_without_whitelist_on_required(db):
    # Without an edge or traversal whitelist the required FK column stays NULL
    # and the database rejects the insert; nothing is persisted or looks saved.
    orphan = GraphUniqueName(name="orphan")
    with pytest.raises(IntegrityError):
        await save_graph([orphan])
    assert orphan.pk is None
    assert orphan._saved_in_db is False
    assert await GraphUniqueName.all().count() == 0
