"""
Identity map support for Tortoise ORM.

This module provides :class:`IdentityScope`, an explicit, nestable identity map
that can be activated per logical request (e.g. per HTTP request or per task).

While a scope is active, every model instance materialized from the database
(regular queries, ``.only()`` partial queries, ``select_related`` and
``prefetch_related``) is keyed by ``(connection name, model class, primary key)``
and the *same* Python object is returned for the same row.

Without an active scope nothing changes: every query keeps constructing fresh
instances, exactly as before.

Rules
-----
- **Partial queries** (``.only()``): fields that have not been loaded yet are
  filled in on the shared instance when a later query provides them. Fields
  that were modified locally (i.e. their current value differs from the value
  loaded from the database) are never overwritten by a merge.
- **Rows without a primary key** (e.g. ``.only()`` that does not include the
  pk) cannot be keyed and are never tracked by the scope.
- **Unmodified fields** may be refreshed with newer database values when a
  later query returns them.
- **save()**: registers the instance in the active scope (if not present) and
  resets the snapshot baseline, so values are considered unmodified again.
- **delete()**: instance-level ``delete()`` evicts the instance from all scopes
  in the active chain. Bulk ``QuerySet.delete()`` does *not* evict (the set of
  affected primary keys is unknown without an extra query); call
  :meth:`IdentityScope.clear` or re-fetch after a bulk delete.
- **refresh_from_db()**: bypasses the identity map for its own query, forces
  the fresh database values onto the instance and resets the snapshot baseline
  for the refreshed fields.
- **Transactions**: the scope is *not* transaction aware. After a transaction
  rollback, cached instances may reflect state that was rolled back. Exit the
  scope (or call :meth:`IdentityScope.clear`) after a rollback.
- **Concurrency**: the active scope is stored in a :class:`contextvars.ContextVar`,
  so concurrent async tasks never share a scope unless they explicitly
  propagate one. Instances from one tenant/request are never reused by another.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tortoise.models import Model

__all__ = ("IdentityScope", "identity_scope", "current_scope")

_current_scope: contextvars.ContextVar[IdentityScope | None] = contextvars.ContextVar(
    "tortoise_identity_scope", default=None
)
_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tortoise_identity_scope_suppressed", default=False
)

_MISSING = object()
_ATOMIC_TYPES = (type(None), bool, int, float, str, bytes)


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, _ATOMIC_TYPES):
        return value
    return deepcopy(value)


def current_scope() -> IdentityScope | None:
    """Return the currently active (and not suppressed) identity scope, if any."""
    scope = _current_scope.get()
    if scope is None or _suppressed.get():
        return None
    return scope


@contextmanager
def _suppress_identity_scope() -> Any:
    """Temporarily disable identity-map participation (used by refresh_from_db)."""
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)


class IdentityScope:
    """
    An explicit, nestable identity map.

    Use as a regular or async context manager::

        with identity_scope():
            a = await Tournament.get(id=1)
            b = await Tournament.get(id=1)
            assert a is b

    Scopes can be nested: the inner scope sees everything registered in outer
    scopes, but instances first loaded in the inner scope are discarded when
    it exits. The active scope is kept in a context variable, so it is
    isolated per async task.
    """

    def __init__(self) -> None:
        # key: (connection_name, model class, pk) -> [instance, snapshot]
        self._entries: dict[tuple[str | None, type[Model], Any], list[Any]] = {}
        self._parent: IdentityScope | None = None
        self._token: contextvars.Token | None = None

    def __enter__(self) -> IdentityScope:
        if self._token is not None:
            raise RuntimeError("An IdentityScope instance cannot be re-entered")
        self._parent = _current_scope.get()
        self._token = _current_scope.set(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._token is not None:
            _current_scope.reset(self._token)
            self._token = None
        self._parent = None
        self.clear()

    async def __aenter__(self) -> IdentityScope:
        return self.__enter__()

    async def __aexit__(self, *exc_info: Any) -> None:
        self.__exit__(*exc_info)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop every instance registered directly in this scope."""
        self._entries.clear()

    def _find_entry(self, key: tuple[str | None, type[Model], Any]) -> list[Any] | None:
        scope: IdentityScope | None = self
        while scope is not None:
            entry = scope._entries.get(key)
            if entry is not None:
                return entry
            scope = scope._parent
        return None

    def _register_or_merge(
        self,
        model: type[Model],
        connection_name: str | None,
        instance: Model,
        loaded_fields: list[str],
    ) -> Model:
        pk = instance.pk
        if pk is None:
            return instance
        key = (connection_name, model, pk)
        entry = self._find_entry(key)
        if entry is None:
            snapshot = {field: _snapshot_value(getattr(instance, field)) for field in loaded_fields}
            self._entries[key] = [instance, snapshot]
            return instance
        existing, snapshot = entry
        for field in loaded_fields:
            new_value = getattr(instance, field)
            old_value = snapshot.get(field, _MISSING)
            if old_value is _MISSING:
                # Field not loaded before: fill it in (e.g. after an .only() query).
                object.__setattr__(existing, field, new_value)
                snapshot[field] = _snapshot_value(new_value)
            elif getattr(existing, field, _MISSING) == old_value:
                # Not modified locally: may be refreshed with the newer DB value.
                if new_value != old_value:
                    object.__setattr__(existing, field, new_value)
                    snapshot[field] = _snapshot_value(new_value)
            # else: locally modified value is never overwritten.
        if existing._partial and all(
            field in snapshot for field in model._meta.fields_db_projection
        ):
            existing._partial = False
        return existing

    def _on_save(self, instance: Model, connection_name: str | None) -> None:
        pk = instance.pk
        if pk is None:
            return
        model = type(instance)
        key = (connection_name, model, pk)
        snapshot = {
            field: _snapshot_value(getattr(instance, field))
            for field in model._meta.fields_db_projection
            if hasattr(instance, field)
        }
        entry = self._find_entry(key)
        if entry is None:
            self._entries[key] = [instance, snapshot]
        elif entry[0] is instance:
            # Saved successfully: current values become the new baseline.
            entry[1].clear()
            entry[1].update(snapshot)
        # A different object with the same identity was saved: the scope keeps
        # the instance it already canonicalized.

    def _on_refresh(self, instance: Model, fields: list[str], connection_name: str | None) -> None:
        key = (connection_name, type(instance), instance.pk)
        entry = self._find_entry(key)
        if entry is not None and entry[0] is instance:
            for field in fields:
                entry[1][field] = _snapshot_value(getattr(instance, field, None))

    def _evict(self, model: type[Model], connection_name: str | None, pk: Any) -> None:
        key = (connection_name, model, pk)
        scope: IdentityScope | None = self
        while scope is not None:
            scope._entries.pop(key, None)
            scope = scope._parent


def identity_scope() -> IdentityScope:
    """
    Create a new :class:`IdentityScope`.

    Usage::

        async with identity_scope():
            ...
    """
    return IdentityScope()
