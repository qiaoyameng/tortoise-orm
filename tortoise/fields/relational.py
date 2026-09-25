from __future__ import annotations

import warnings
from collections.abc import AsyncGenerator, Generator, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

from pypika_tortoise.queries import Table

from tortoise.exceptions import ConfigurationError, NoValuesFetched, OperationalError
from tortoise.fields.base import CASCADE, SET_NULL, Field, OnDelete

if TYPE_CHECKING:  # pragma: nocoverage
    from tortoise.backends.base.client import BaseDBAsyncClient
    from tortoise.models import Model
    from tortoise.queryset import Q, QuerySet

MODEL = TypeVar("MODEL", bound="Model")


class _NoneAwaitable:
    __slots__ = ()

    def __await__(self) -> Generator[None, None, None]:
        yield None

    def __bool__(self) -> bool:
        return False


NoneAwaitable = _NoneAwaitable()


class ReverseRelation(Generic[MODEL]):
    """
    Relation container for :func:`.ForeignKeyField`.
    """

    def __init__(
        self,
        remote_model: type[MODEL],
        relation_field: str,
        instance: Model,
        from_field: str,
    ) -> None:
        self.remote_model = remote_model
        self.relation_field = relation_field
        self.instance = instance
        self.from_field = from_field
        self._fetched = False
        self._custom_query = False
        self.related_objects: list[MODEL] = []

    @property
    def _query(self) -> QuerySet[MODEL]:
        if not self.instance._saved_in_db:
            raise OperationalError(
                "This objects hasn't been instanced, call .save() before calling related queries"
            )
        return self.remote_model.filter(
            **{self.relation_field: getattr(self.instance, self.from_field)}
        )

    def __contains__(self, item: Any) -> bool:
        self._raise_if_not_fetched()
        return item in self.related_objects

    def __iter__(self) -> Iterator[MODEL]:
        self._raise_if_not_fetched()
        return self.related_objects.__iter__()

    def __len__(self) -> int:
        self._raise_if_not_fetched()
        return len(self.related_objects)

    def __bool__(self) -> bool:
        self._raise_if_not_fetched()
        return bool(self.related_objects)

    def __getitem__(self, item: int) -> MODEL:
        self._raise_if_not_fetched()
        return self.related_objects[item]

    def __await__(self) -> Generator[Any, None, list[MODEL]]:
        return self._query.__await__()

    async def __aiter__(self) -> AsyncGenerator[MODEL, None]:
        if not self._fetched:
            self._set_result_for_query(await self)
        for val in self.related_objects:
            yield val

    def filter(self, *args: Q, **kwargs: Any) -> QuerySet[MODEL]:
        """
        Returns a QuerySet with related elements filtered by args/kwargs.
        """
        return self._query.filter(*args, **kwargs)

    def all(self) -> QuerySet[MODEL]:
        """
        Returns a QuerySet with all related elements.
        """
        return self._query

    def order_by(self, *orderings: str) -> QuerySet[MODEL]:
        """
        Returns a QuerySet related elements in order.
        """
        return self._query.order_by(*orderings)

    def limit(self, limit: int) -> QuerySet[MODEL]:
        """
        Returns a QuerySet with at most «limit» related elements.
        """
        return self._query.limit(limit)

    def offset(self, offset: int) -> QuerySet[MODEL]:
        """
        Returns a QuerySet with all related elements offset by «offset».
        """
        return self._query.offset(offset)

    async def create(self, using_db: BaseDBAsyncClient | None = None, **kwargs: Any) -> MODEL:
        """
        Create a related record in the DB and returns the object, automatically setting the
        foreign key relationship to the parent instance.

        .. code-block:: python3

            tournament = await Tournament.create(name="...")
            event = await tournament.events.create(...)

        Equivalent to:

        .. code-block:: python3

            tournament = await Tournament.create(name="...")
            event = await Event.create(tournament=tournament, ...)

        :param using_db: Specific DB connection to use instead of default bound.
        :param kwargs: Model parameters for the new object.
        :raises OperationalError: If parent instance is not saved to the database.
        """
        if not self.instance._saved_in_db:
            raise OperationalError(
                "This objects hasn't been instanced, call .save() before calling related queries"
            )

        # Inject foreign key relationship automatically
        kwargs[self.relation_field] = getattr(self.instance, self.from_field)

        # Call remote model's create method
        return await self.remote_model.create(using_db=using_db, **kwargs)

    def _set_result_for_query(self, sequence: list[MODEL], attr: str | None = None) -> None:
        self._fetched = True
        self.related_objects = sequence
        if attr:
            setattr(self.instance, attr, sequence)

    def _raise_if_not_fetched(self) -> None:
        if not self._fetched:
            raise NoValuesFetched(
                "No values were fetched for this relation, first use .fetch_related()"
            )


class ManyToManyRelation(ReverseRelation[MODEL]):
    """
    Many-to-many relation container for :func:`.ManyToManyField`.
    """

    def __init__(self, instance: Model, m2m_field: ManyToManyFieldInstance[MODEL]) -> None:
        super().__init__(m2m_field.related_model, m2m_field.related_name, instance, "pk")
        self.field = m2m_field
        self.instance = instance

    def _get_through_model(self) -> type[Model] | None:
        """
        Returns the registered model whose table matches the through table of this
        relation, or ``None`` if no such model exists.
        """
        from tortoise import Tortoise
        from tortoise.context import get_current_context

        ctx = get_current_context()
        if ctx is not None and ctx._inited and ctx.apps is not None:
            apps = ctx.apps
        else:
            apps = Tortoise.apps
        if not apps:
            return None
        for model in apps.get_models_iterable():
            if (
                model._meta.db_table == self.field.through
                and model._meta.schema == self.field.through_schema
            ):
                return model
        return None

    def _normalize_through_defaults(
        self,
        through_defaults: dict[str, Any] | Iterable[dict[str, Any] | None],
        count: int,
    ) -> list[dict[str, Any]]:
        """
        Normalizes ``through_defaults`` to a list of dicts, one per instance to add.

        A single dict is broadcast to all instances; a sequence of dicts (or ``None``
        entries) is matched positionally with the instances.
        """
        if isinstance(through_defaults, dict):
            return [dict(through_defaults) for _ in range(count)]
        defaults_list = list(through_defaults)
        if len(defaults_list) != count:
            raise ConfigurationError(
                f"through_defaults has {len(defaults_list)} entries, "
                f"but {count} instance(s) were passed to add()"
            )
        normalized: list[dict[str, Any]] = []
        for defaults in defaults_list:
            if defaults is None:
                normalized.append({})
            elif isinstance(defaults, dict):
                normalized.append(dict(defaults))
            else:
                raise ConfigurationError(
                    "through_defaults entries must be dicts or None, "
                    f"got {type(defaults).__name__}"
                )
        return normalized

    def _resolve_through_defaults_fields(
        self, through_model: type[Model], defaults_list: list[dict[str, Any]]
    ) -> tuple[list[str], list[str], list[Field]]:
        """
        Validates the keys of ``through_defaults`` against the through model and
        returns the keys, DB columns and fields to write, in a stable order.

        Empty dicts (and ``None`` entries) mean "no extra values for this
        instance"; their missing keys are written as NULL. All non-empty dicts
        must provide the same set of fields.
        """
        keys: tuple[str, ...] | None = None
        keys_list: list[str] = []
        for defaults in defaults_list:
            if not defaults:
                continue
            if keys is None:
                keys = tuple(defaults.keys())
            elif set(defaults.keys()) != set(keys):
                raise ConfigurationError(
                    "All non-empty through_defaults dicts of one add() call "
                    "must provide the same set of fields"
                )
        columns: list[str] = []
        fields: list[Field] = []
        for key in keys or ():
            field = through_model._meta.fields_map.get(key)
            column = through_model._meta.fields_db_projection.get(key)
            if field is None or column is None:
                raise ConfigurationError(
                    f'"{key}" is not a concrete field of through model '
                    f'"{through_model.__name__}"'
                )
            if column in (self.field.backward_key, self.field.forward_key):
                raise ConfigurationError(
                    f'Cannot set relation key "{key}" via through_defaults, '
                    "it is managed by the many-to-many relation itself"
                )
            keys_list.append(key)
            columns.append(column)
            fields.append(field)
        return keys_list, columns, fields

    async def add(
        self,
        *instances: MODEL,
        using_db: BaseDBAsyncClient | None = None,
        through_defaults: dict[str, Any] | Iterable[dict[str, Any] | None] | None = None,
    ) -> None:
        """
        Adds one or more of ``instances`` to the relation.

        If it is already added, it will be silently ignored.

        Extra values for an explicitly configured through model (a registered model
        whose table is the relation's through table) can be supplied via
        ``through_defaults``: either a single dict applied to every added instance,
        or a sequence of dicts (``None`` for no extra values) matched positionally
        with ``instances``. All keys and values are validated before anything is
        written, so an invalid value aborts the whole call without persisting any
        of the new relations. Relations that already exist are left untouched
        (their through values are never updated). Callable values are evaluated
        once per relation that is actually inserted; existing relations are skipped
        without evaluating them.

        :param using_db: Specific DB connection to use instead of default bound.
        :param through_defaults: Extra values for the through model, either one dict
            for all instances or one dict per instance.
        :raises OperationalError: If Object to add is not saved.
        :raises ConfigurationError: If ``through_defaults`` is malformed, refers to
            unknown fields, or no through model is registered for the through table.
        :raises ValidationError: If a ``through_defaults`` value is invalid for its
            through model field.
        """
        if not instances:
            return
        if not self.instance._saved_in_db:
            raise OperationalError(f"You should first call .save() on {self.instance}")
        through_model: type[Model] | None = None
        extra_keys: list[str] = []
        extra_columns: list[str] = []
        extra_fields: list[Field] = []
        if through_defaults is None:
            defaults_list: list[dict[str, Any]] | None = None
        else:
            defaults_list = self._normalize_through_defaults(through_defaults, len(instances))
            through_model = self._get_through_model()
            if through_model is None:
                raise ConfigurationError(
                    "through_defaults requires a model registered for the through "
                    f'table "{self.field.through}"'
                )
            extra_keys, extra_columns, extra_fields = self._resolve_through_defaults_fields(
                through_model, defaults_list
            )
        db = using_db or self.remote_model._meta.db
        pk_formatting_func = type(self.instance)._meta.pk.to_db_value
        related_pk_formatting_func = type(instances[0])._meta.pk.to_db_value
        pk_b = pk_formatting_func(self.instance.pk, self.instance)
        pks_f: list = []
        for instance_to_add in instances:
            if not instance_to_add._saved_in_db:
                raise OperationalError(f"You should first call .save() on {instance_to_add}")
            pk_f = related_pk_formatting_func(instance_to_add.pk, instance_to_add)
            pks_f.append(pk_f)
        through_table = Table(self.field.through, schema=self.field.through_schema)
        backward_key, forward_key = self.field.backward_key, self.field.forward_key
        backward_field, forward_field = through_table[backward_key], through_table[forward_key]
        select_query = (
            db.query_class.from_(through_table).where(backward_field == pk_b).select(forward_key)
        )
        criterion = forward_field == pks_f[0] if len(pks_f) == 1 else forward_field.isin(pks_f)
        select_query = select_query.where(criterion)

        _, already_existing_relations_raw = await db.execute_query(
            *select_query.get_parameterized_sql()
        )
        already_existing_forward_pks = {
            related_pk_formatting_func(r[forward_key], self.instance)
            for r in already_existing_relations_raw
        }

        if pks_f_to_insert := set(pks_f) - already_existing_forward_pks:
            extra_column_fields = [through_table[column] for column in extra_columns]
            query = db.query_class.into(through_table).columns(
                forward_field, backward_field, *extra_column_fields
            )
            defaults_by_pk: dict = {}
            if defaults_list is not None:
                for pk_f, defaults in zip(pks_f, defaults_list):
                    defaults_by_pk.setdefault(pk_f, defaults)
            for pk_f in pks_f_to_insert:
                row: list = [pk_f, pk_b]
                if defaults_list is not None and through_model is not None:
                    defaults = defaults_by_pk[pk_f]
                    for key, field in zip(extra_keys, extra_fields):
                        value = defaults.get(key)
                        if callable(value):
                            value = value()
                        row.append(field.to_db_value(value, through_model))
                query = query.insert(*row)
            await db.execute_query(*query.get_parameterized_sql())

    async def clear(self, using_db: BaseDBAsyncClient | None = None) -> None:
        """
        Clears ALL relations.
        """
        await self._remove_or_clear(using_db=using_db)

    async def remove(self, *instances: MODEL, using_db: BaseDBAsyncClient | None = None) -> None:
        """
        Removes one or more of ``instances`` from the relation.

        :raises OperationalError: remove() was called with no instances.
        """
        if not instances:
            raise OperationalError("remove() called on no instances")
        await self._remove_or_clear(instances, using_db)

    async def _remove_or_clear(
        self,
        instances: tuple[MODEL, ...] | None = None,
        using_db: BaseDBAsyncClient | None = None,
    ) -> None:
        db = using_db or self.remote_model._meta.db
        through_table = Table(self.field.through, schema=self.field.through_schema)
        pk_formatting_func = type(self.instance)._meta.pk.to_db_value

        condition = through_table[self.field.backward_key] == pk_formatting_func(
            self.instance.pk, self.instance
        )
        if instances:
            related_pk_formatting_func = type(instances[0])._meta.pk.to_db_value
            if len(instances) == 1:
                condition &= through_table[self.field.forward_key] == related_pk_formatting_func(
                    instances[0].pk, instances[0]
                )
            else:
                condition &= through_table[self.field.forward_key].isin(
                    [related_pk_formatting_func(i.pk, i) for i in instances]
                )
        query = db.query_class.from_(through_table).where(condition).delete()
        await db.execute_query(*query.get_parameterized_sql())


class RelationalField(Field[MODEL]):
    has_db_field = False

    def __init__(
        self,
        related_model: type[MODEL],
        to_field: str | None = None,
        db_constraint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.related_model: type[MODEL] = related_model
        self.to_field: str = to_field  # type: ignore
        self.to_field_instance: Field = None  # type: ignore
        self.db_constraint = db_constraint

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: type[Model]) -> RelationalField[MODEL]: ...

        @overload
        def __get__(self, instance: Model, owner: type[Model]) -> MODEL: ...

        def __get__(
            self, instance: Model | None, owner: type[Model]
        ) -> RelationalField[MODEL] | MODEL: ...

        def __set__(self, instance: Model, value: MODEL) -> None: ...

    def describe(self, serializable: bool) -> dict:
        desc = super().describe(serializable)
        desc["db_constraint"] = self.db_constraint
        del desc["db_column"]
        return desc

    @classmethod
    def validate_model_name(cls, model_name: str | type[Model]) -> None:
        if not isinstance(model_name, str):
            model_class: type[Model] = model_name
            try:
                model_class._meta
            except AttributeError:
                raise ConfigurationError(
                    f"{cls.__name__}({model_name!r}) is invalid. model_name must be string or type[tortoise.models.Model]"
                ) from None
        elif len(model_name.split(".")) != 2:
            field_type = cls.__name__.replace("Instance", "")
            raise ConfigurationError(f'{field_type} accepts model name in format "app.Model"')


class ForeignKeyFieldInstance(RelationalField[MODEL]):
    def __init__(
        self,
        model_name: type[Model] | str,
        related_name: str | None | Literal[False] = None,
        on_delete: OnDelete = CASCADE,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, **kwargs)  # type:ignore[arg-type]
        self.validate_model_name(model_name)
        self.model_name = model_name
        self.related_name = related_name
        if on_delete not in set(OnDelete):
            raise ConfigurationError(
                "on_delete can only be CASCADE, RESTRICT, SET_NULL, SET_DEFAULT or NO_ACTION"
            )
        if on_delete == SET_NULL and not bool(kwargs.get("null")):
            raise ConfigurationError("If on_delete is SET_NULL, then field must have null=True set")
        self.on_delete = on_delete

    def describe(self, serializable: bool) -> dict:
        desc = super().describe(serializable)
        desc["raw_field"] = self.source_field
        desc["on_delete"] = str(self.on_delete)
        return desc

    def deconstruct(self) -> tuple[str, list[Any], dict[str, Any]]:
        path, args, kwargs = super().deconstruct()
        # After Apps._init_relations(), source_field no longer holds the column name: it
        # holds the name of the generated `<field>_id` backing field. The column name the
        # user asked for was moved onto that backing field, so read it back from there.
        # Without this, makemigrations writes the backing field name as the column name
        # and a migrated schema disagrees with generate_schemas().
        model = getattr(self, "model", None)
        if model is not None and self.source_field:
            backing_field = model._meta.fields_map.get(self.source_field)
            if backing_field is not None and backing_field.source_field:
                kwargs["source_field"] = backing_field.source_field
        return path, args, kwargs


class BackwardFKRelation(RelationalField[MODEL]):
    def __init__(
        self,
        field_type: type[MODEL],
        relation_field: str,
        relation_source_field: str,
        null: bool,
        description: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(field_type, null=null, **kwargs)
        self.relation_field: str = relation_field
        self.relation_source_field: str = relation_source_field
        self.description: str | None = description


class OneToOneFieldInstance(ForeignKeyFieldInstance[MODEL]):
    def __init__(
        self,
        model_name: type[MODEL] | str,
        related_name: str | None | Literal[False] = None,
        on_delete: OnDelete = CASCADE,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, related_name, on_delete, unique=True, **kwargs)

    def deconstruct(self) -> tuple[str, list[Any], dict[str, Any]]:
        path, args, kwargs = super().deconstruct()
        # unique=True is set implicitly by OneToOneField.__init__
        kwargs.pop("unique", None)
        return path, args, kwargs


class BackwardOneToOneRelation(BackwardFKRelation[MODEL]):
    pass


class ManyToManyFieldInstance(RelationalField[MODEL]):
    field_type = ManyToManyRelation

    def __init__(
        self,
        model_name: type[MODEL] | str,
        through: str | None = None,
        forward_key: str | None = None,
        backward_key: str = "",
        related_name: str = "",
        on_delete: OnDelete = CASCADE,
        field_type: type[MODEL] = None,  # type: ignore
        unique: bool = True,
        **kwargs: Any,
    ) -> None:
        # TODO: rename through to through_table
        # TODO: add through to use a Model
        if "create_unique_index" in kwargs:
            warnings.warn(
                "Parameter `create_unique_index` is deprecated! Use `unique` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            unique = kwargs.pop("create_unique_index")
        super().__init__(field_type, unique=unique, **kwargs)
        self.validate_model_name(model_name)
        self.model_name = model_name
        self.related_name: str = related_name
        if not forward_key:
            model_class_name = (
                model_name.split(".")[1]  # e.g.: 'models.Users' -> 'Users'
                if isinstance(model_name, str)
                else model_name.__name__
            )
            forward_key = f"{model_class_name.lower()}_id"
        self.forward_key: str = forward_key
        self.backward_key: str = backward_key
        self.through: str = through  # type: ignore
        self.through_schema: str | None = None
        self._generated: bool = False
        self.on_delete = on_delete

    def describe(self, serializable: bool) -> dict:
        desc = super().describe(serializable)
        if isinstance(self.model_name, str):
            model_name = self.model_name
        else:
            model: type[MODEL] = self.model_name
            model_name = f"{model._meta.app}.{model.__name__}"
        desc["model_name"] = model_name
        desc["related_name"] = self.related_name
        desc["forward_key"] = self.forward_key
        desc["backward_key"] = self.backward_key
        desc["through"] = self.through
        desc["on_delete"] = str(self.on_delete)
        desc["_generated"] = self._generated
        return desc


@overload
def OneToOneField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    *,
    null: Literal[True],
    **kwargs: Any,
) -> OneToOneNullableRelation[MODEL]: ...


@overload
def OneToOneField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    null: Literal[False] = False,
    **kwargs: Any,
) -> OneToOneRelation[MODEL]: ...


def OneToOneField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    null: bool = False,
    **kwargs: Any,
) -> OneToOneRelation[MODEL] | OneToOneNullableRelation[MODEL]:
    """
    OneToOne relation field.

    This field represents a foreign key relation to another model.

    See :ref:`one_to_one` for usage information.

    You must provide the following:

    ``to``:
        The related model or name of the related model in a :samp:`'{app}.{model}'` format.

    The following is optional:

    ``related_name``:
        The attribute name on the related model to reverse resolve the foreign key.
    ``on_delete``:
        One of:
            ``field.CASCADE``:
                Indicate that the model should be cascade deleted if related model gets deleted.
            ``field.RESTRICT``:
                Indicate that the related model delete will be restricted as long as a
                foreign key points to it.
            ``field.SET_NULL``:
                Resets the field to NULL in case the related model gets deleted.
                Can only be set if field has ``null=True`` set.
            ``field.SET_DEFAULT``:
                Resets the field to ``default`` value in case the related model gets deleted.
                Can only be set is field has a ``default`` set.
            ``field.NO_ACTION``:
                Take no action.
    ``to_field``:
        The attribute name on the related model to establish foreign key relationship.
        If not set, pk is used
    ``db_constraint``:
        Controls whether or not a constraint should be created in the database for this foreign key.
        The default is True, and that’s almost certainly what you want; setting this to False can be very bad for data integrity.
    """

    return OneToOneFieldInstance(
        to, related_name, on_delete, db_constraint=db_constraint, null=null, **kwargs
    )


@overload
def ForeignKeyField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    *,
    null: Literal[True],
    **kwargs: Any,
) -> ForeignKeyNullableRelation[MODEL]: ...


@overload
def ForeignKeyField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    null: Literal[False] = False,
    **kwargs: Any,
) -> ForeignKeyRelation[MODEL]: ...


def ForeignKeyField(
    to: type[MODEL] | str,
    related_name: str | None | Literal[False] = None,
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    null: bool = False,
    **kwargs: Any,
) -> ForeignKeyRelation[MODEL] | ForeignKeyNullableRelation[MODEL]:
    """
    ForeignKey relation field.

    This field represents a foreign key relation to another model.

    See :ref:`foreign_key` for usage information.

    You must provide the following:

    ``to``:
        The related model or name of the related model in a :samp:`'{app}.{model}'` format.

    The following is optional:

    ``related_name``:
        The attribute name on the related model to reverse resolve the foreign key.
    ``on_delete``:
        One of:
            ``field.CASCADE``:
                Indicate that the model should be cascade deleted if related model gets deleted.
            ``field.RESTRICT``:
                Indicate that the related model delete will be restricted as long as a
                foreign key points to it.
            ``field.SET_NULL``:
                Resets the field to NULL in case the related model gets deleted.
                Can only be set if field has ``null=True`` set.
            ``field.SET_DEFAULT``:
                Resets the field to ``default`` value in case the related model gets deleted.
                Can only be set is field has a ``default`` set.
            ``field.NO_ACTION``:
                Take no action.
    ``to_field``:
        The attribute name on the related model to establish foreign key relationship.
        If not set, pk is used
    ``db_constraint``:
        Controls whether or not a constraint should be created in the database for this foreign key.
        The default is True, and that’s almost certainly what you want; setting this to False can be very bad for data integrity.
    """

    return ForeignKeyFieldInstance(
        to, related_name, on_delete, db_constraint=db_constraint, null=null, **kwargs
    )


def ManyToManyField(
    to: type[MODEL] | str,
    through: str | None = None,
    forward_key: str | None = None,
    backward_key: str = "",
    related_name: str = "",
    on_delete: OnDelete = CASCADE,
    db_constraint: bool = True,
    unique: bool = True,
    **kwargs: Any,
) -> ManyToManyRelation[MODEL]:
    """
    ManyToMany relation field.

    This field represents a many-to-many between this model and another model.

    See :ref:`many_to_many` for usage information.

    You must provide the following:

    ``to``:
        The related model or name of the related model in a :samp:`'{app}.{model}'` format.

    The following is optional:

    ``through``:
        The DB table that represents the through table.
        The default is normally safe.
    ``forward_key``:
        The forward lookup key on the through table.
        The default is normally safe.
    ``backward_key``:
        The backward lookup key on the through table.
        The default is normally safe.
    ``related_name``:
        The attribute name on the related model to reverse resolve the many to many.
    ``db_constraint``:
        Controls whether or not a constraint should be created in the database for this foreign key.
        The default is True, and that’s almost certainly what you want; setting this to False can be very bad for data integrity.
    ``on_delete``:
        One of:
            ``field.CASCADE``:
                Indicate that the model should be cascade deleted if related model gets deleted.
            ``field.RESTRICT``:
                Indicate that the related model delete will be restricted as long as a
                foreign key points to it.
            ``field.SET_NULL``:
                Resets the field to NULL in case the related model gets deleted.
                Can only be set if field has ``null=True`` set.
            ``field.SET_DEFAULT``:
                Resets the field to ``default`` value in case the related model gets deleted.
                Can only be set is field has a ``default`` set.
            ``field.NO_ACTION``:
                Take no action.
    ``unique``:
        Controls whether or not a unique index should be created in the database to speed up select queries.
        The default is True. If you want to allow repeat records, set this to False.
    """
    return ManyToManyFieldInstance(  # type: ignore
        to,
        through,
        forward_key,
        backward_key,
        related_name,
        on_delete=on_delete,
        db_constraint=db_constraint,
        unique=unique,
        **kwargs,
    )


OneToOneNullableRelation = OneToOneFieldInstance[MODEL] | None
"""
Type hint for the result of accessing the :func:`.OneToOneField` field in the model
when obtained model can be nullable.
"""

OneToOneRelation = OneToOneFieldInstance[MODEL]
"""
Type hint for the result of accessing the :func:`.OneToOneField` field in the model.
"""

ForeignKeyNullableRelation = ForeignKeyFieldInstance[MODEL] | None
"""
Type hint for the result of accessing the :func:`.ForeignKeyField` field in the model
when obtained model can be nullable.
"""

ForeignKeyRelation = ForeignKeyFieldInstance[MODEL]
"""
Type hint for the result of accessing the :func:`.ForeignKeyField` field in the model.
"""
