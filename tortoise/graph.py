"""Transactional whole-object-graph persistence (:func:`save_graph`)."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from tortoise.exceptions import ConfigurationError, ParamsError
from tortoise.transactions import in_transaction

if TYPE_CHECKING:
    from tortoise.backends.base.client import BaseDBAsyncClient
    from tortoise.fields.relational import ForeignKeyFieldInstance
    from tortoise.models import Model

_MISSING = object()


class Edge(NamedTuple):
    """An explicit relation edge for :func:`save_graph`.

    :param source: The model instance holding the relation field.
    :param field: Name of a ``ForeignKeyField``/``OneToOneField`` on *source*.
    :param target: The related instance (possibly unsaved), or ``None`` to
        clear a nullable relation.
    """

    source: Model
    field: str
    target: Model | None


# ---------------------------------------------------------------------------
# Traversal whitelist normalisation
# ---------------------------------------------------------------------------


def _normalise_whitelist(
    traverse: Iterable[type[Model] | str | tuple[type[Model] | str, str]] | None,
) -> dict[type[Model], set[str]]:
    """Turn the user supplied *traverse* argument into ``{model: {field}}``."""

    if traverse is None:
        return {}

    result: dict[type[Model], set[str]] = defaultdict(set)

    def _resolve_model(model_ref: type[Model] | str) -> type[Model]:
        if isinstance(model_ref, type):
            return model_ref
        if not isinstance(model_ref, str):
            raise ParamsError(
                "traverse entries must be a model class, a 'Model.field' string"
                " or a (model, field) tuple"
            )
        from tortoise import Tortoise

        apps = Tortoise.apps
        if apps is None:
            raise ConfigurationError("Tortoise ORM is not initialised")

        if "." in model_ref:
            app_label, model_name = model_ref.split(".", 1)
            return apps.get_model(app_label, model_name)

        # Bare model name: resolve across apps (ambiguous names are an error).
        matches = [
            model
            for model in apps.get_models_iterable()
            if model.__name__ == model_ref
        ]
        if not matches:
            raise ConfigurationError(f"No model named {model_ref!r} is registered")
        if len(matches) > 1:
            raise ConfigurationError(
                f"Model name {model_ref!r} is ambiguous, qualify it as"
                " 'app.Model'"
            )
        return matches[0]

    for entry in traverse:
        if isinstance(entry, str):
            model_ref, sep, field = entry.rpartition(".")
            if not sep:
                raise ParamsError(
                    f"traverse string {entry!r} must look like 'Model.field'"
                )
            result[_resolve_model(model_ref)].add(field)
        elif isinstance(entry, tuple) and len(entry) == 2:
            model = _resolve_model(entry[0])
            field = entry[1]
            if not isinstance(field, str):
                raise ParamsError("traverse tuple field must be a string")
            result[model].add(field)
        else:
            model = _resolve_model(entry)  # type: ignore[arg-type]
            meta = model._meta
            result[model].update(set(meta.fk_fields) | set(meta.o2o_fields))

    # Validate field names up-front.
    for model, fields in result.items():
        valid = set(model._meta.fk_fields) | set(model._meta.o2o_fields)
        unknown = fields - valid
        if unknown:
            raise ConfigurationError(
                f"{model.__name__} has no forward relation field(s): "
                + ", ".join(sorted(unknown))
            )

    return dict(result)


def _validate_fk_field(model: type[Model], field: str) -> ForeignKeyFieldInstance:
    from tortoise.fields.relational import (
        ForeignKeyFieldInstance,
        OneToOneFieldInstance,
    )

    fk = model._meta.fields_map.get(field)
    if not isinstance(fk, (ForeignKeyFieldInstance, OneToOneFieldInstance)):
        raise ConfigurationError(
            f"{model.__name__}.{field} is not a ForeignKey/OneToOne relation field"
        )
    return fk


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _related_value(instance: Model, field: str) -> tuple[bool, Any]:
    """Read the cached ``_<field>`` relation without hitting the DB.

    Returns ``(present, value)``; *present* is ``False`` when no relation
    object was assigned on the instance.
    """

    try:
        return True, getattr(instance, f"_{field}")
    except AttributeError:
        return False, None


def _collect_graph(
    roots: list[Model],
    explicit_edges: list[Edge],
    whitelist: dict[type[Model], set[str]],
) -> tuple[
    dict[int, Model],
    dict[tuple[int, str], Edge],
    set[tuple[int, str]],
]:
    """Collect nodes/edges reachable through explicit edges and the whitelist.

    Returns ``(nodes, edges)``: *nodes* maps ``id(obj)`` to every seen
    instance (saved and unsaved), *edges* maps ``(id(source), field)`` to the
    winning edge (explicit edges win over traversed relations).
    """

    nodes: dict[int, Model] = {}
    edges: dict[tuple[int, str], Edge] = {}
    explicit_keys: set[tuple[int, str]] = set()

    def _register(instance: Model) -> None:
        nodes.setdefault(id(instance), instance)

    for root in roots:
        _register(root)

    for edge in explicit_edges:
        source, field, target = edge
        fk = _validate_fk_field(type(source), field)
        if target is not None:
            if type(target) is not fk.related_model:
                raise ParamsError(
                    f"Edge {type(source).__name__}.{field} expects"
                    f" {fk.related_model.__name__}, got {type(target).__name__}"
                )
            _register(target)
        elif not fk.null:
            raise ParamsError(
                f"Cannot clear non-nullable relation {type(source).__name__}.{field}"
            )
        key = (id(source), field)
        if key in edges:
            raise ParamsError(
                f"Duplicate edge for {type(source).__name__}.{field}"
            )
        edges[key] = edge
        explicit_keys.add(key)
        _register(source)

    # Whitelisted traversal, starting from unsaved roots only. Saved
    # instances are the reference boundary and are never expanded.
    pending: deque[Model] = deque(r for r in roots if not r._saved_in_db)
    seen: set[int] = {id(obj) for obj in pending}
    while pending:
        instance = pending.popleft()
        allowed = whitelist.get(type(instance))
        if not allowed:
            continue
        for field in allowed:
            fk = _validate_fk_field(type(instance), field)
            present, target = _related_value(instance, field)
            if not present:
                continue
            key = (id(instance), field)
            if key not in edges:
                if target is not None and type(target) is not fk.related_model:
                    raise ParamsError(
                        f"Relation {type(instance).__name__}.{field} expects"
                        f" {fk.related_model.__name__}, got {type(target).__name__}"
                    )
                if target is None and not fk.null:
                    raise ParamsError(
                        f"Cannot clear non-nullable relation"
                        f" {type(instance).__name__}.{field}"
                    )
                edges[key] = Edge(instance, field, target)
            target = edges[key].target
            if target is not None and id(target) not in seen:
                seen.add(id(target))
                _register(target)
                if not target._saved_in_db:
                    pending.append(target)

    # Explicit edges have stricter rules: their source must be one of the
    # supplied (unsaved) roots. Traversal-discovered edges, by contrast,
    # legitimately originate on nodes reached through the whitelist.
    root_ids = {id(r) for r in roots}
    for key in explicit_keys:
        edge = edges[key]
        if edge.source._saved_in_db:
            raise ConfigurationError(
                "save_graph cannot write relation"
                f" {type(edge.source).__name__}.{edge.field} on an already"
                " saved instance"
            )
        if id(edge.source) not in root_ids:
            raise ConfigurationError(
                f"Explicit edge source {type(edge.source).__name__} is not part"
                " of the objects collection; add it explicitly"
            )

    return nodes, edges, explicit_keys


# ---------------------------------------------------------------------------
# Ordering / cycle analysis
# ---------------------------------------------------------------------------


def _plan_inserts(
    nodes: dict[int, Model],
    edges: dict[tuple[int, str], Edge],
) -> tuple[list[Model], dict[tuple[int, str], Edge]]:
    """Produce insert order and the set of deferred nullable back-edges."""

    unsaved: dict[int, Model] = {
        oid: obj for oid, obj in nodes.items() if not obj._saved_in_db
    }
    unsaved_ids = set(unsaved)

    # Outgoing unsaved-to-unsaved edges grouped by source.
    outgoing: dict[int, list[Edge]] = defaultdict(list)
    for edge in edges.values():
        source_id = id(edge.source)
        if source_id not in unsaved_ids or edge.target is None:
            continue
        target_id = id(edge.target)
        if target_id not in unsaved_ids:
            continue
        outgoing[source_id].append(edge)

    # A non-nullable cycle is impossible to insert: diagnose it first.
    non_null_cycle = _find_non_nullable_cycle(outgoing, unsaved)
    if non_null_cycle is not None:
        raise ConfigurationError(_format_cycle(non_null_cycle))

    # Kahn over every unsaved->unsaved edge. Nullable back-edges of cycles are
    # removed from the constraint graph and deferred until both rows exist.
    deps: dict[int, set[int]] = {oid: set() for oid in unsaved_ids}
    for source_id, source_edges in outgoing.items():
        for edge in source_edges:
            deps[source_id].add(id(edge.target))

    deferred: dict[tuple[int, str], Edge] = {}
    order: list[Model] = []
    inserted: set[int] = set()
    remaining = set(unsaved_ids)
    while remaining:
        ready = sorted(
            (oid for oid in remaining if deps[oid] <= inserted),
            key=lambda oid: unsaved[oid].__class__.__name__,
        )
        if not ready:
            # Everything left belongs to cycles. Every remaining cycle must
            # contain a nullable edge (checked above); defer all nullable
            # edges of one stuck source that point at remaining nodes until
            # that source becomes ready, which breaks the cycle open.
            progressed = False
            for source_id in sorted(
                remaining, key=lambda oid: unsaved[oid].__class__.__name__
            ):
                for edge in outgoing[source_id]:
                    target_id = id(edge.target)
                    fk = _validate_fk_field(type(edge.source), edge.field)
                    if target_id in remaining and fk.null:
                        deps[source_id].discard(target_id)
                        deferred[(source_id, edge.field)] = edge
                        progressed = True
                if deps[source_id] <= inserted:
                    break
            if not progressed:
                # Defensive: _find_non_nullable_cycle should have caught this.
                raise ConfigurationError(
                    "Cannot order inserts due to a non-nullable cycle: "
                    + ", ".join(
                        f"{unsaved[oid].__class__.__name__}#{oid}"
                        for oid in remaining
                    )
                )
            continue

        for oid in ready:
            order.append(unsaved[oid])
            inserted.add(oid)
            remaining.remove(oid)

    return order, deferred


def _find_non_nullable_cycle(
    outgoing: dict[int, list[Edge]],
    unsaved: dict[int, Model],
) -> list[Edge] | None:
    """Return a closed chain of non-nullable edges, or ``None``."""

    color: dict[int, int] = {}  # 0 = unseen, 1 = on stack, 2 = done
    stack_edges: list[Edge] = []
    stack_nodes: list[int] = []

    def visit(node_id: int) -> list[Edge] | None:
        color[node_id] = 1
        stack_nodes.append(node_id)
        for edge in outgoing.get(node_id, ()):
            fk = _validate_fk_field(type(edge.source), edge.field)
            if fk.null:
                continue
            target_id = id(edge.target)
            if target_id not in unsaved:
                continue
            state = color.get(target_id, 0)
            if state == 1:
                start = stack_nodes.index(target_id)
                return stack_edges[start:] + [edge]
            if state == 0:
                stack_edges.append(edge)
                found = visit(target_id)
                if found is not None:
                    return found
                stack_edges.pop()
        color[node_id] = 2
        stack_nodes.pop()
        return None

    for oid in sorted(unsaved, key=lambda x: unsaved[x].__class__.__name__):
        if color.get(oid, 0) == 0:
            cycle = visit(oid)
            if cycle is not None:
                return cycle
    return None


def _format_cycle(cycle_edges: list[Edge]) -> str:
    start = type(cycle_edges[0].source).__name__
    steps = [f".{edge.field}->{type(edge.target).__name__}" for edge in cycle_edges]
    return (
        "Non-nullable relation cycle cannot be saved: "
        f"{start}{''.join(steps)} (back to {start})"
    )


# ---------------------------------------------------------------------------
# State snapshot / restore
# ---------------------------------------------------------------------------


def _snapshot(nodes: dict[int, Model]) -> dict[str, Any]:
    unsaved = [obj for obj in nodes.values() if not obj._saved_in_db]

    node_state: list[tuple[Model, Any, bool, bool]] = []
    for obj in unsaved:
        node_state.append((obj, obj.pk, obj._saved_in_db, obj._custom_generated_pk))

    return {"nodes": node_state}


def _restore(snapshot: dict[str, Any]) -> None:
    for obj, pk, saved, custom_pk in snapshot["nodes"]:
        obj.pk = pk
        obj._saved_in_db = saved
        obj._custom_generated_pk = custom_pk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Sentinel stored in the "previous value" snapshots for genuinely missing
# attributes (distinguishes unset from explicitly ``None``).
class _EdgesToApply:
    """Caches edge values on instances so they can be reverted on rollback."""

    __slots__ = ("records",)

    def __init__(self) -> None:
        # (source, cache_attr, backing_attr, had_cache, old_cache,
        #  had_backing, old_backing)
        self.records: list[tuple[Model, str, str, bool, Any, bool, Any]] = []

    def apply(self, source: Model, field: str, target: Model | None) -> None:
        fk = _validate_fk_field(type(source), field)
        cache_attr = f"_{field}"
        backing_attr = fk.source_field

        had_cache = hasattr(source, cache_attr)
        old_cache = getattr(source, cache_attr, _MISSING)
        had_backing = hasattr(source, backing_attr)
        old_backing = getattr(source, backing_attr, _MISSING)
        self.records.append(
            (
                source,
                cache_attr,
                backing_attr,
                had_cache,
                old_cache,
                had_backing,
                old_backing,
            )
        )

        if target is None:
            object.__setattr__(source, backing_attr, None)
            object.__setattr__(source, cache_attr, None)
        else:
            object.__setattr__(source, backing_attr, target.pk)
            object.__setattr__(source, cache_attr, target)

    def restore_all(self) -> None:
        for (
            source,
            cache_attr,
            backing_attr,
            had_cache,
            old_cache,
            had_backing,
            old_backing,
        ) in reversed(self.records):
            if had_cache:
                object.__setattr__(source, cache_attr, old_cache)
            else:
                if hasattr(source, cache_attr):
                    delattr(source, cache_attr)
            if had_backing:
                object.__setattr__(source, backing_attr, old_backing)
            elif hasattr(source, backing_attr):
                delattr(source, backing_attr)


async def save_graph(
    objects: Iterable[Model],
    edges: Iterable[Edge | tuple[Model, str, Model | None]] | None = None,
    *,
    traverse: Iterable[type[Model] | str | tuple[type[Model] | str, str]] | None = None,
    using_db: BaseDBAsyncClient | None = None,
) -> list[Model]:
    """Persist a graph of related new objects in a single transaction.

    ``edges`` may point at instances that do not have a primary key yet; the
    targets are inserted first according to foreign key dependencies. Shared
    (diamond) objects are inserted only once, already saved instances act as
    reference boundaries and are never written, and traversal of pre-assigned
    relations is restricted to the *traverse* whitelist so that unrelated
    objects are never saved implicitly.

    Nullable cycles are broken by inserting the back-edge column as ``NULL``
    and updating it once both rows exist. Non-nullable cycles are rejected
    with a path diagnostic *before* anything is written to the database.

    On any failure the transaction is rolled back and every touched instance
    is restored to its pre-call state (allocated primary keys and
    ``_saved_in_db`` flags included), so in-memory objects never look
    persisted when the graph did not fully persist.

    :param objects: Root instances to save. Unsaved instances are inserted;
        already saved instances are used as references only.
    :param edges: Explicit relation edges as :class:`Edge` or
        ``(source, field_name, target)`` triples.
    :param traverse: Optional whitelist of forward relations to follow from
        unsaved instances. Accepts model classes (all relations),
        ``"Model.field"`` strings or ``(model, "field")`` tuples.
    :param using_db: Specific DB connection to use. All touched models must
        share it.
    :raises ConfigurationError: Invalid graph shape or a non-nullable cycle.
    :raises ParamsError: Malformed arguments/edges.
    """

    from tortoise.models import Model

    roots = list(objects)
    for root in roots:
        if not isinstance(root, Model):
            raise ParamsError("objects must contain Model instances")

    normalised_edges: list[Edge] = []
    for edge in edges or ():
        if isinstance(edge, Edge):
            normalised_edges.append(edge)
        elif (
            isinstance(edge, tuple)
            and len(edge) == 3
            and isinstance(edge[1], str)
        ):
            normalised_edges.append(Edge(edge[0], edge[1], edge[2]))
        else:
            raise ParamsError(
                "edges must be Edge or (source, field_name, target) triples"
            )

    whitelist = _normalise_whitelist(traverse)
    nodes, graph_edges, _ = _collect_graph(roots, normalised_edges, whitelist)
    order, deferred = _plan_inserts(nodes, graph_edges)

    # All writes must go to one connection so a single transaction covers the
    # whole graph.
    db = using_db
    if db is None:
        for obj in order:
            candidate = obj._choose_db(True)
            if db is None:
                db = candidate
            elif candidate.connection_name != db.connection_name:
                raise ConfigurationError(
                    "save_graph requires all objects to share one database"
                    f" connection; {db.connection_name} vs"
                    f" {candidate.connection_name}"
                )
    if db is None and not order:
        return []

    # Validate edge target model types once more against the related model,
    # and prepare values to apply/revert.
    applied = _EdgesToApply()
    snapshot = _snapshot(nodes)

    # Edges by source, restricted to unsaved sources.
    edges_by_source: dict[int, list[Edge]] = defaultdict(list)
    for edge in graph_edges.values():
        if not edge.source._saved_in_db:
            edges_by_source[id(edge.source)].append(edge)

    try:
        async with in_transaction(db.connection_name):
            inserted: list[Model] = []
            for instance in order:
                for edge in edges_by_source.get(id(instance), ()):
                    if (id(instance), edge.field) in deferred:
                        # Leave the column empty for now; backfill later.
                        applied.apply(edge.source, edge.field, None)
                    else:
                        applied.apply(edge.source, edge.field, edge.target)
                await instance.save(using_db=db)
                inserted.append(instance)

            # Backfill nullable cycle edges. Both endpoints now have PKs.
            for (source_id, field), edge in deferred.items():
                source = nodes[source_id]
                fk = _validate_fk_field(type(source), field)
                object.__setattr__(source, fk.source_field, edge.target.pk)
                object.__setattr__(source, f"_{field}", edge.target)
                executor = db.executor_class(model=type(source), db=db)
                await executor.execute_update(source, (fk.source_field,))

        return inserted
    except BaseException:
        _restore(snapshot)
        applied.restore_all()
        raise
