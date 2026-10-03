from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from tortoise.exceptions import GraphCycleError, ParamsError
from tortoise.models import Model
from tortoise.transactions import in_transaction

if TYPE_CHECKING:  # pragma: nocoverage
    from tortoise.backends.base.client import BaseDBAsyncClient

_SENTINEL = object()


class _Edge(NamedTuple):
    source: Model
    field_name: str
    target: Model
    source_field: str
    nullable: bool


def _format_cycle(cycle: list[_Edge]) -> str:
    parts = [f"{type(edge.source).__name__}.{edge.field_name}" for edge in cycle]
    parts.append(type(cycle[-1].target).__name__)
    return " -> ".join(parts)


def _find_cycle(
    node_ids: list[int], adjacency: dict[int, list[tuple[int, _Edge]]]
) -> list[_Edge] | None:
    """Return the edges of one cycle in the directed graph, or None if acyclic."""
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(node_ids, white)
    for start in node_ids:
        if color[start] != white:
            continue
        color[start] = gray
        stack = [(start, iter(adjacency.get(start, ())))]
        path_nodes = [start]
        path_edges: list[_Edge] = []
        while stack:
            node, candidates = stack[-1]
            advanced = False
            for target_id, edge in candidates:
                if color[target_id] == gray:
                    index = path_nodes.index(target_id)
                    return path_edges[index:] + [edge]
                if color[target_id] == white:
                    color[target_id] = gray
                    stack.append((target_id, iter(adjacency.get(target_id, ()))))
                    path_nodes.append(target_id)
                    path_edges.append(edge)
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                color[node] = black
                path_nodes.pop()
                if path_edges and len(path_edges) == len(path_nodes):
                    path_edges.pop()
    return None


async def save_graph(
    objects: Iterable[Model],
    edges: Iterable[tuple[Model, str, Model]],
    using_db: "BaseDBAsyncClient | None" = None,
) -> None:
    """
    Saves a graph of related new objects in a single transaction.

    ``objects`` is the explicit whitelist of instances to insert. Instances that
    are already saved act as reference boundaries: they are never re-inserted and
    never modified. Objects reachable through relations but absent from
    ``objects`` are never saved implicitly.

    ``edges`` is a list of ``(source, field_name, target)`` tuples describing
    forward FK/O2O relations to establish. ``target`` may be an unsaved instance
    as long as it is part of ``objects``; its primary key is resolved once it has
    been inserted.

    Shared objects are inserted exactly once. Acyclic parts are inserted in
    foreign-key dependency order. Cycles are only supported when they can be
    broken by a nullable FK, which is inserted as NULL first and backfilled
    afterwards; otherwise a :class:`GraphCycleError` with the offending path is
    raised before anything is written.

    On any failure the transaction is rolled back and the primary keys and saved
    state assigned during this call are restored, so in-memory objects do not
    look persisted.

    :param objects: The instances to insert (already saved ones are boundaries).
    :param edges: ``(source, field_name, target)`` relation edges to establish.
    :param using_db: Optional DB connection; defaults to the default connection.

    :raises ParamsError: If an edge is malformed or references an unsaved object
        that is not part of ``objects``.
    :raises GraphCycleError: If a relation cycle can not be broken by a nullable
        foreign key. Raised before any database write.
    """
    # --- Normalize and validate the object whitelist ---
    unique_objects: list[Model] = []
    seen: set[int] = set()
    for obj in objects:
        if not isinstance(obj, Model):
            raise ParamsError(f"save_graph objects must be Model instances, got {obj!r}")
        if id(obj) not in seen:
            seen.add(id(obj))
            unique_objects.append(obj)

    object_ids = {id(obj) for obj in unique_objects}
    by_id = {id(obj): obj for obj in unique_objects}

    # --- Normalize and validate edges ---
    parsed_edges: list[_Edge] = []
    for edge in edges:
        try:
            source, field_name, target = edge
        except (TypeError, ValueError):
            raise ParamsError(
                f"Edges must be (source, field_name, target) tuples, got {edge!r}"
            ) from None
        if id(source) not in object_ids:
            raise ParamsError(
                f"Edge source {source!r} is not part of the objects collection; "
                "save_graph only saves explicitly whitelisted objects"
            )
        if not isinstance(target, Model):
            raise ParamsError(f"Edge target must be a Model instance, got {target!r}")
        meta = source._meta
        if field_name not in meta.fk_fields and field_name not in meta.o2o_fields:
            raise ParamsError(
                f"Field '{field_name}' is not a FK/O2O relation on {type(source).__name__}"
            )
        if source._saved_in_db:
            raise ParamsError(
                f"Edge source {source!r} is already saved; saved objects are reference "
                "boundaries and are not modified by save_graph"
            )
        if not target._saved_in_db and id(target) not in object_ids:
            raise ParamsError(
                f"Edge target {target!r} is unsaved and not part of the objects collection; "
                "add it to objects or save it first"
            )
        field_object = meta.fields_map[field_name]
        raw_field = meta.fields_map[field_object.source_field]
        parsed_edges.append(
            _Edge(
                source=source,
                field_name=field_name,
                target=target,
                source_field=field_object.source_field,
                nullable=bool(raw_field.null),
            )
        )

    pending = [obj for obj in unique_objects if not obj._saved_in_db]
    if not pending:
        return

    # --- Break cycles by deferring nullable edges; reject unbreakable cycles ---
    internal_edges = [edge for edge in parsed_edges if not edge.target._saved_in_db]
    deferred: list[_Edge] = []
    active = list(internal_edges)
    pending_ids = [id(obj) for obj in pending]
    while True:
        adjacency: dict[int, list[tuple[int, _Edge]]] = {}
        for edge in active:
            adjacency.setdefault(id(edge.source), []).append((id(edge.target), edge))
        cycle = _find_cycle(pending_ids, adjacency)
        if cycle is None:
            break
        nullable_edges = [edge for edge in cycle if edge.nullable]
        if not nullable_edges:
            raise GraphCycleError(
                "Can not save graph: non-nullable relation cycle "
                f"{_format_cycle(cycle)}. Make at least one relation on the cycle "
                "nullable so it can be inserted as NULL and backfilled."
            )
        chosen = nullable_edges[0]
        deferred.append(chosen)
        active = [edge for edge in active if edge is not chosen]

    # --- Topologically sort pending objects by FK dependency (stable) ---
    indegree = dict.fromkeys(pending_ids, 0)
    dependents: dict[int, list[int]] = {}
    for edge in active:
        source_id, target_id = id(edge.source), id(edge.target)
        if source_id == target_id:
            continue  # self-loop edges are always deferred above
        indegree[source_id] += 1
        dependents.setdefault(target_id, []).append(source_id)
    order: list[Model] = []
    available = [node_id for node_id in pending_ids if indegree[node_id] == 0]
    while available:
        node_id = available.pop(0)
        order.append(by_id[node_id])
        for dependent_id in dependents.get(node_id, ()):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                available.append(dependent_id)

    # Group the edges that are applied before the source insert, by source.
    deferred_ids = {id(edge) for edge in deferred}
    immediate_by_source: dict[int, list[_Edge]] = {}
    for edge in parsed_edges:
        if id(edge) not in deferred_ids:
            immediate_by_source.setdefault(id(edge.source), []).append(edge)

    # --- Insert everything in one transaction, tracking state for rollback ---
    saved_this_call: list[tuple[Model, Any]] = []
    fk_restore: list[tuple[Model, str, Any, str, Any]] = []

    def apply_edge(edge: _Edge) -> None:
        source = edge.source
        cache_attr = f"_{edge.field_name}"
        old_raw = getattr(source, edge.source_field)
        old_cache = getattr(source, cache_attr, _SENTINEL)
        fk_restore.append((source, edge.source_field, old_raw, cache_attr, old_cache))
        setattr(source, edge.field_name, edge.target)

    connection = using_db._in_transaction() if using_db is not None else in_transaction()
    try:
        async with connection as conn:
            for obj in order:
                for edge in immediate_by_source.get(id(obj), ()):
                    apply_edge(edge)
                old_pk = obj.pk
                await obj.save(using_db=conn)
                saved_this_call.append((obj, old_pk))
            for edge in deferred:
                apply_edge(edge)
                await edge.source.save(using_db=conn, update_fields=[edge.source_field])
    except BaseException:
        for obj, old_pk in reversed(saved_this_call):
            setattr(obj, obj._meta.pk_attr, old_pk)
            obj._saved_in_db = False
        for source, source_field, old_raw, cache_attr, old_cache in reversed(fk_restore):
            object.__setattr__(source, source_field, old_raw)
            if old_cache is _SENTINEL:
                source.__dict__.pop(cache_attr, None)
            else:
                object.__setattr__(source, cache_attr, old_cache)
        raise
