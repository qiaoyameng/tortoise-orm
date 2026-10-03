from __future__ import annotations

import contextvars
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tortoise.backends.base.client import BaseDBAsyncClient
    from tortoise.models import Model


IdentityKey = tuple[str, Any, Any]


def _dirty_fields(instance: Model) -> set[str] | None:
    fields = getattr(instance, "_identity_dirty_fields", None)
    return fields if isinstance(fields, set) else None


def mark_field_dirty(instance: Model, field_name: str) -> None:
    scope = get_current_identity_scope()
    if scope is None:
        return

    fields = _dirty_fields(instance)
    if fields is None:
        fields = set()
        object.__setattr__(instance, "_identity_dirty_fields", fields)
    fields.add(field_name)


def clear_dirty_fields(instance: Model, field_names: set[str] | None = None) -> None:
    fields = _dirty_fields(instance)
    if fields is None:
        return
    if field_names is None:
        fields.clear()
    else:
        fields.difference_update(field_names)


class IdentityScope:
    """A task-local cache of persisted model instances keyed by their identity."""

    def __init__(self, parent: IdentityScope | None = None) -> None:
        self._objects: dict[IdentityKey, Model] = {}
        self._parent = parent
        self._token: contextvars.Token[IdentityScope | None] | None = None

    def __enter__(self) -> IdentityScope:
        self._token = _current_identity_scope.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _current_identity_scope.reset(self._token)
            self._token = None

    async def __aenter__(self) -> IdentityScope:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_value, traceback)

    def _find(self, key: IdentityKey) -> tuple[IdentityScope, Model] | None:
        instance = self._objects.get(key)
        if instance is not None:
            return self, instance
        if self._parent is not None:
            return self._parent._find(key)
        return None

    def get(
        self,
        db: BaseDBAsyncClient | str,
        model: type[Model],
        pk: Any,
    ) -> Model | None:
        if pk is None:
            return None
        connection_name = db if isinstance(db, str) else db.connection_name
        found = self._find((connection_name, model, pk))
        return found[1] if found is not None else None

    def merge(
        self,
        db: BaseDBAsyncClient | str,
        model: type[Model],
        values: dict[str, Any],
        instance: Model | None = None,
        *,
        custom_fields: list[str] | None = None,
    ) -> Model:
        connection_name = db if isinstance(db, str) else db.connection_name
        pk_value = self._get_pk_value(model, values)

        if instance is None:
            if pk_value is not None:
                found = self._find((connection_name, model, pk_value))
                if found is not None:
                    target_scope, instance = found
                else:
                    instance = model._init_from_db(**values)
                    self._objects[(connection_name, model, pk_value)] = instance
                    target_scope = self
            else:
                instance = model._init_from_db(**values)
                target_scope = None
        else:
            pk_value = instance.pk
            key = (connection_name, model, pk_value)
            found = self._find(key) if pk_value is not None else None
            if found is not None and found[1] is not instance:
                _, instance = found
            elif pk_value is not None:
                self._objects[key] = instance
            target_scope = self

        self._load_values(instance, model, values, custom_fields=custom_fields)
        if target_scope is not None and pk_value is not None:
            target_scope._objects[(connection_name, model, pk_value)] = instance
        return instance

    def register(self, db: BaseDBAsyncClient | str, instance: Model) -> None:
        pk = instance.pk
        if pk is None:
            return
        connection_name = db if isinstance(db, str) else db.connection_name
        self._objects[(connection_name, type(instance), pk)] = instance

    def evict(
        self,
        db: BaseDBAsyncClient | str,
        model: type[Model],
        pk: Any,
    ) -> None:
        if pk is None:
            return
        connection_name = db if isinstance(db, str) else db.connection_name
        key = (connection_name, model, pk)
        self._objects.pop(key, None)
        if self._parent is not None:
            self._parent.evict(db, model, pk)

    def evict_model(self, db: BaseDBAsyncClient | str, model: type[Model]) -> None:
        connection_name = db if isinstance(db, str) else db.connection_name
        prefix = (connection_name, model)
        for key in list(self._objects):
            if key[:2] == prefix:
                self._objects.pop(key, None)
        if self._parent is not None:
            self._parent.evict_model(db, model)

    def evict_connection(self, connection_name: str) -> None:
        for key in list(self._objects):
            if key[0] == connection_name:
                self._objects.pop(key, None)
        if self._parent is not None:
            self._parent.evict_connection(connection_name)

    @staticmethod
    def _get_pk_value(model: type[Model], values: dict[str, Any]) -> Any:
        pk_attr = model._meta.pk_attr
        if pk_attr in values:
            return values[pk_attr]
        return values.get(model._meta.fields_db_projection.get(pk_attr, pk_attr))

    @staticmethod
    def _load_values(
        instance: Model,
        model: type[Model],
        values: dict[str, Any],
        *,
        custom_fields: list[str] | None = None,
    ) -> None:
        dirty = _dirty_fields(instance) or set()
        projection = model._meta.fields_db_projection
        reverse_projection = model._meta.fields_db_projection_reverse
        for key, value in values.items():
            field_name = reverse_projection.get(key, key)
            if field_name in dirty:
                continue
            if field_name in projection:
                field = model._meta.fields_map[field_name]
                value = field.to_python_value(value)
            elif key != field_name:
                continue
            object.__setattr__(instance, field_name, value)
            dirty.discard(field_name)

        for field_name in custom_fields or ():
            if field_name not in values or field_name in dirty:
                continue
            object.__setattr__(instance, field_name, values[field_name])
            dirty.discard(field_name)

        loaded_fields = {
            reverse_projection.get(key, key)
            for key in values
            if reverse_projection.get(key, key) in model._meta.fields_map
        }
        if loaded_fields >= set(model._meta.fields_db_projection) and not dirty.intersection(
            model._meta.fields_db_projection
        ):
            object.__setattr__(instance, "_partial", False)
        elif getattr(instance, "_partial", False):
            object.__setattr__(instance, "_partial", True)

    def _load_related(
        self,
        db: BaseDBAsyncClient | str,
        instance: Model,
        field_name: str,
        related: Model | None,
    ) -> None:
        field = instance._meta.fields_map[field_name]
        backing_name = getattr(field, "source_field", None)
        dirty = _dirty_fields(instance) or set()
        if field_name in dirty or (backing_name is not None and backing_name in dirty):
            return

        object.__setattr__(instance, f"_{field_name}", related)
        if backing_name is not None:
            object.__setattr__(instance, backing_name, related.pk if related is not None else None)
            dirty.discard(backing_name)
        dirty.discard(field_name)


_current_identity_scope: contextvars.ContextVar[IdentityScope | None] = contextvars.ContextVar(
    "tortoise_identity_scope", default=None
)


def identity_scope() -> IdentityScope:
    parent = _current_identity_scope.get()
    scope = IdentityScope(parent)
    return scope


def get_current_identity_scope() -> IdentityScope | None:
    return _current_identity_scope.get()


def load_related(
    db: BaseDBAsyncClient | str,
    instance: Model,
    field_name: str,
    related: Model | None,
) -> None:
    scope = get_current_identity_scope()
    if scope is None:
        field = instance._meta.fields_map[field_name]
        object.__setattr__(instance, f"_{field_name}", related)
        backing_name = getattr(field, "source_field", None)
        if backing_name is not None:
            object.__setattr__(
                instance, backing_name, related.pk if related is not None else None
            )
        return
    scope._load_related(db, instance, field_name, related)


def register_instance(db: BaseDBAsyncClient | str, instance: Model) -> None:
    scope = get_current_identity_scope()
    if scope is not None:
        scope.register(db, instance)


def evict_instance(db: BaseDBAsyncClient | str, instance: Model) -> None:
    scope = get_current_identity_scope()
    if scope is not None:
        scope.evict(db, type(instance), instance.pk)


def evict_model(db: BaseDBAsyncClient | str, model: type[Model]) -> None:
    scope = get_current_identity_scope()
    if scope is not None:
        scope.evict_model(db, model)


def evict_connection(connection_name: str) -> None:
    scope = get_current_identity_scope()
    if scope is not None:
        scope.evict_connection(connection_name)


__all__ = [
    "IdentityScope",
    "identity_scope",
]
