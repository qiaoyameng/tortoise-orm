from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from pypika_tortoise import Case as PypikaCase
from pypika_tortoise import Field as PypikaField
from pypika_tortoise import Order, SqlContext, Table
from pypika_tortoise.analytics import CURRENT_ROW, Following, Preceding
from pypika_tortoise.functions import AggregateFunction, DistinctOptionFunction
from pypika_tortoise.terms import (
    AnalyticFunction,
    ArithmeticExpression,
    Criterion,
    Term,
    ValueWrapper,
    WindowFrameAnalyticFunction,
)
from pypika_tortoise.terms import Function as PypikaFunction
from pypika_tortoise.utils import format_alias_sql

from tortoise.exceptions import FieldError, OperationalError, UnSupportedError
from tortoise.fields.base import Field
from tortoise.fields.data import JSONField
from tortoise.fields.relational import RelationalField
from tortoise.filters import FilterInfoDict
from tortoise.query_utils import (
    QueryModifier,
    TableCriterionTuple,
    get_joins_for_related_field,
    resolve_field_json_path,
    resolve_nested_field,
)

if TYPE_CHECKING:  # pragma: nocoverage
    from pypika_tortoise.queries import Selectable

    from tortoise.models import Model
    from tortoise.queryset import AwaitableQuery


@dataclass(frozen=True)
class ResolveContext:
    model: type[Model]
    table: Table
    annotations: dict[str, Any]
    custom_filters: dict[str, FilterInfoDict]


@dataclass
class ResolveResult:
    term: Term
    joins: list[TableCriterionTuple] = dataclass_field(default_factory=list)
    output_field: Field | None = None


class Expression:
    """
    Parent class for expressions
    """

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        raise NotImplementedError()


class Value(Expression):
    """
    Wrapper for a value that should be used as a term in a query.
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        return ResolveResult(term=ValueWrapper(self.value))


class Connector(Enum):
    add = "add"
    sub = "sub"
    mul = "mul"
    div = "truediv"
    pow = "pow"
    mod = "mod"


class CombinedExpression(Expression):
    def __init__(self, left: Expression, connector: Connector, right: Any) -> None:
        self.left = left
        self.connector = connector
        self.right = right if isinstance(right, Expression) else Value(right)

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        left = self.left.resolve(resolve_context)
        right = self.right.resolve(resolve_context)
        left_output_field, right_output_field = left.output_field, right.output_field  # type: ignore

        if (
            left_output_field
            and right_output_field
            and type(left_output_field) is not type(right_output_field)
        ):
            raise FieldError("Cannot use arithmetic expression between different field type")

        operator_func = getattr(operator, self.connector.value)
        return ResolveResult(
            term=operator_func(left.term, right.term),
            joins=list(set(left.joins + right.joins)),  # dedup joins
            output_field=right_output_field or left_output_field,
        )


class F(Expression):
    """
    F() can be used to reference a model field, field of a related model, annotation or
    an attribute of a JSON field. It can be used in the following ways:

    - as a field reference, e.g. F("id")
    - as a related field reference, e.g. F("related_field__field") will return the value of the field
    of the related model.
    - as a JSON field reference, e.g. F("json_field__attribute") will return the value of the "attribute"
    property of the JSON field value. The reference can be nested, e.g. F("json_field__attribute__subattribute")
    - as a JSON field array element reference, e.g. F("json_field__0") will return the first element of the array.

    :param name: The name of the field to reference.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        term: Term
        joins: list[TableCriterionTuple] = []
        output_field = None

        main_name_part, __, rest_name_parts = self.name.partition("__")
        if main_name_part in resolve_context.model._meta.fetch_fields:
            # field in the format of "related_field__field" or "related_field__another_rel_field__field"
            term, joins, output_field = resolve_nested_field(
                resolve_context.model, resolve_context.table, self.name
            )
        elif (
            rest_name_parts
            and main_name_part in resolve_context.model._meta.fields_map
            and isinstance(resolve_context.model._meta.fields_map[main_name_part], JSONField)
        ):
            # Accessing a JSON field, e.g. F("json_field__a__b")
            key_parts = [
                int(item) if item.isdigit() else str(item) for item in rest_name_parts.split("__")
            ]
            term = resolve_field_json_path(
                PypikaField(resolve_context.model._meta.fields_db_projection[main_name_part]),
                key_parts,
            )
        elif self.name in resolve_context.annotations:
            # reference to another annotation, e.g. M.annotate(f1=...).annotate(f2=F("f1")).values('field')
            annotation = resolve_context.annotations[self.name]
            if isinstance(annotation, Term):
                term = annotation
            else:
                term = annotation.resolve(resolve_context).term
        else:
            # a regular model field, e.g. F("id")
            try:
                meta = resolve_context.model._meta
                term = PypikaField(meta.fields_db_projection[self.name])

                if (output_field := meta.fields_map.get(self.name, None)) and (
                    func := output_field.get_for_dialect(
                        meta.db.capabilities.dialect, "function_cast"
                    )
                ):
                    term = func(output_field, term)
            except KeyError:
                raise FieldError(
                    f"There is no non-virtual field {self.name} on Model {resolve_context.model.__name__}"
                ) from None
        return ResolveResult(term=term, output_field=output_field, joins=joins)

    def _combine(self, other: Any, connector: Connector, right_hand: bool) -> CombinedExpression:
        if not isinstance(other, Expression):
            other = Value(other)

        if right_hand:
            return CombinedExpression(other, connector, self)
        return CombinedExpression(self, connector, other)

    def __neg__(self) -> CombinedExpression:
        return self._combine(-1, Connector.mul, False)

    def __add__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.add, False)

    def __sub__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.sub, False)

    def __mul__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.mul, False)

    def __truediv__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.div, False)

    def __mod__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.mod, False)

    def __pow__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.pow, False)

    def __radd__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.add, True)

    def __rsub__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.sub, True)

    def __rmul__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.mul, True)

    def __rtruediv__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.div, True)

    def __rmod__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.mod, True)

    def __rpow__(self, other) -> CombinedExpression:
        return self._combine(other, Connector.pow, True)


class Subquery(Term):
    def __init__(self, query: AwaitableQuery) -> None:
        super().__init__()
        self.query = query

    def get_sql(self, ctx: SqlContext) -> str:
        self.query._choose_db_if_not_chosen()
        self.query._make_query()
        return self.query.query.get_parameterized_sql(ctx)[0]

    def as_(self, alias: str) -> Selectable:  # type: ignore
        self.query._choose_db_if_not_chosen()
        self.query._make_query()
        return self.query.query.as_(alias)


class RawSQL(Term):
    def __init__(self, sql: str) -> None:
        super().__init__()
        self.sql = sql

    def get_sql(self, ctx: SqlContext) -> str:
        if ctx.with_alias:
            return format_alias_sql(sql=self.sql, alias=self.alias, ctx=ctx)
        return self.sql


class Q:
    """
    Q Expression container.
    Q Expressions are a useful tool to compose a query from many small parts.

    :param join_type: Is the join an AND or OR join type?
    :param args: Inner ``Q`` expressions that you want to wrap.
    :param kwargs: Filter statements that this Q object should encapsulate.
    """

    __slots__ = (
        "children",
        "filters",
        "join_type",
        "_is_negated",
    )

    AND = "AND"
    OR = "OR"

    def __init__(self, *args: Q, join_type: str = AND, **kwargs: Any) -> None:
        if args and kwargs:
            newarg = Q(join_type=join_type, **kwargs)
            args = (newarg,) + args
            kwargs = {}
        if not all(isinstance(node, Q) for node in args):
            raise OperationalError("All ordered arguments must be Q nodes")
        #: Contains the sub-Q's that this Q is made up of
        self.children: tuple[Q, ...] = args
        #: Contains the filters applied to this Q
        self.filters: dict[str, FilterInfoDict] = kwargs
        if join_type not in {self.AND, self.OR}:
            raise OperationalError("join_type must be AND or OR")
        #: Specifies if this Q does an AND or OR on its children
        self.join_type = join_type
        self._is_negated = False

    def __and__(self, other: Q) -> Q:
        """
        Returns a binary AND of Q objects, use ``AND`` operator.

        :raises OperationalError: AND operation requires a Q node
        """
        if not isinstance(other, Q):
            raise OperationalError("AND operation requires a Q node")
        return Q(self, other, join_type=self.AND)

    def __or__(self, other: Q) -> Q:
        """
        Returns a binary OR of Q objects, use ``OR`` operator.

        :raises OperationalError: OR operation requires a Q node
        """
        if not isinstance(other, Q):
            raise OperationalError("OR operation requires a Q node")
        return Q(self, other, join_type=self.OR)

    def __invert__(self) -> Q:
        """
        Returns a negated instance of the Q object, use ``~`` operator.
        """
        q = Q(*self.children, join_type=self.join_type, **self.filters)
        q.negate()
        return q

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Q):
            return False
        return (
            self.children == other.children
            and self.join_type == other.join_type
            and self.filters == other.filters
        )

    def __bool__(self) -> bool:
        if self.filters:
            return True
        return any(self.children)

    def negate(self) -> None:
        """
        Negates the current Q object. (mutation)
        """
        self._is_negated = not self._is_negated

    def _resolve_nested_filter(
        self, resolve_context: ResolveContext, key: str, value: Any, table: Table
    ) -> QueryModifier:
        related_field_name, __, forwarded_fields = key.partition("__")
        related_field = cast(
            RelationalField, resolve_context.model._meta.fields_map[related_field_name]
        )
        required_joins = get_joins_for_related_field(table, related_field, related_field_name)
        q = Q(**{forwarded_fields: value})
        modifier = q.resolve(
            ResolveContext(
                model=related_field.related_model,
                table=required_joins[-1][0],
                annotations=resolve_context.annotations,
                custom_filters=resolve_context.custom_filters,
            )
        )
        return QueryModifier(joins=required_joins) & modifier

    def _resolve_custom_kwarg(
        self, resolve_context: ResolveContext, key: str, value: Any, table: Table
    ) -> QueryModifier:
        having_info = resolve_context.custom_filters[key]
        annotation = resolve_context.annotations[having_info["field"]]
        if isinstance(annotation, Term):
            annotation_info = ResolveResult(term=annotation)
        else:
            annotation_info = annotation.resolve(resolve_context)

        operator = having_info["operator"]
        overridden_operator = (
            resolve_context.model._meta.db.executor_class.get_overridden_filter_func(
                filter_func=operator, filter_info=having_info
            )
        )
        if overridden_operator:
            operator = overridden_operator
        if annotation_info.term.is_aggregate:
            modifier = QueryModifier(having_criterion=operator(annotation_info.term, value))
        else:
            modifier = QueryModifier(where_criterion=operator(annotation_info.term, value))
        return modifier

    def _process_filter_kwarg(
        self, model: type[Model], key: str, value: Any, table: Table
    ) -> tuple[Criterion, tuple[Table, Criterion] | None]:
        join = None

        if value is None and f"{key}__isnull" in model._meta.filters:
            filter_info = model._meta.get_filter(f"{key}__isnull")
            value = True
        else:
            filter_info = model._meta.get_filter(key)

        field_object = None
        if "table" in filter_info:
            # join the table
            join = (
                filter_info["table"],
                table[model._meta.db_pk_column]
                == filter_info["table"][filter_info["backward_key"]],
            )
            if "value_encoder" in filter_info:
                value = filter_info["value_encoder"](value, model)
            table = filter_info["table"]
        elif not isinstance(value, Term):
            field_object = model._meta.fields_map[filter_info["field"]]
            value = (
                filter_info["value_encoder"](value, model, field_object)
                if "value_encoder" in filter_info
                else field_object.to_db_value(value, model)
            )
        op = filter_info["operator"]
        term: Term = table[filter_info.get("source_field", filter_info["field"])]
        if field_object is not None:
            func = field_object.get_for_dialect(
                model._meta.db.capabilities.dialect, "function_cast"
            )
            if func is not None:
                term = func(field_object, term)
        criterion = op(term, value)
        return criterion, join

    def _resolve_regular_kwarg(
        self, resolve_context: ResolveContext, key: str, value: Any, table: Table
    ) -> QueryModifier:
        if (
            key not in resolve_context.model._meta.filters
            and key.split("__")[0] in resolve_context.model._meta.fetch_fields
        ):
            modifier = self._resolve_nested_filter(resolve_context, key, value, table)
        else:
            criterion, join = self._process_filter_kwarg(resolve_context.model, key, value, table)
            joins = [join] if join else []
            modifier = QueryModifier(where_criterion=criterion, joins=joins)
        return modifier

    def _get_actual_filter_params(
        self, resolve_context: ResolveContext, key: str, value: Table | FilterInfoDict
    ) -> tuple[str, Any]:
        filter_key = key
        if (
            key in resolve_context.model._meta.fk_fields
            or key in resolve_context.model._meta.o2o_fields
        ):
            field_object = resolve_context.model._meta.fields_map[key]
            filter_key = cast(str, field_object.source_field)
            filter_value = getattr(value, "pk", value)
        elif key in resolve_context.model._meta.m2m_fields:
            filter_value = getattr(value, "pk", value)
        elif (
            key.split("__")[0] in resolve_context.model._meta.fetch_fields
            or key in resolve_context.custom_filters
            or key in resolve_context.model._meta.filters
        ):
            filter_value = value
        else:
            allowed = sorted(
                resolve_context.model._meta.fields
                | resolve_context.model._meta.fetch_fields
                | set(resolve_context.custom_filters)
            )
            raise FieldError(f"Unknown filter param '{key}'. Allowed base values are {allowed}")

        if isinstance(filter_value, Expression):
            filter_value = filter_value.resolve(resolve_context).term

        return filter_key, filter_value

    def _resolve_kwargs(self, resolve_context: ResolveContext) -> QueryModifier:
        modifier = QueryModifier()
        for raw_key, raw_value in self.filters.items():
            key, value = self._get_actual_filter_params(resolve_context, raw_key, raw_value)
            if key in resolve_context.custom_filters:
                filter_modifier = self._resolve_custom_kwarg(
                    resolve_context, key, value, resolve_context.table
                )
            else:
                filter_modifier = self._resolve_regular_kwarg(
                    resolve_context, key, value, resolve_context.table
                )

            if self.join_type == self.AND:
                modifier &= filter_modifier
            else:
                modifier |= filter_modifier
        if self._is_negated:
            modifier = ~modifier
        return modifier

    def _resolve_children(self, resolve_context: ResolveContext) -> QueryModifier:
        modifier = QueryModifier()
        for node in self.children:
            node_modifier = node.resolve(resolve_context)
            if self.join_type == self.AND:
                modifier &= node_modifier
            else:
                modifier |= node_modifier

        if self._is_negated:
            modifier = ~modifier
        return modifier

    def resolve(
        self,
        resolve_context: ResolveContext,
    ) -> QueryModifier:
        """
        Resolves the logical Q chain into the parts of a SQL statement.

        :param model: The Model this Q Expression should be resolved on.
        :param table: ``pypika_tortoise.Table`` to keep track of the virtual SQL table
            (to allow self referential joins)
        """
        if self.filters:
            return self._resolve_kwargs(resolve_context)
        return self._resolve_children(resolve_context)


class Function(Expression):
    """
    Function/Aggregate base.

    :param field: Field name
    :param default_values: Extra parameters to the function.

    .. attribute:: database_func
        :annotation: pypika_tortoise.terms.Function

        The pypika function this represents.

    .. attribute:: populate_field_object
        :annotation: bool = False

        Enable populate_field_object where we want to try and preserve the field type.
    """

    __slots__ = ("field", "field_object", "default_values")

    database_func: type[PypikaFunction] = PypikaFunction
    # Enable populate_field_object where we want to try and preserve the field type.
    populate_field_object = False

    def __init__(
        self, field: str | F | CombinedExpression | Function | Term, *default_values: Any
    ) -> None:
        self.field = field
        self.field_object: Field | None = None
        self.default_values = default_values

    def _get_function_field(self, field: Term | str, *default_values) -> PypikaFunction:
        return self.database_func(field, *default_values)  # type:ignore[arg-type]

    def _resolve_nested_field(self, resolve_context: ResolveContext, field: str) -> ResolveResult:
        term, joins, output_field = resolve_nested_field(
            resolve_context.model, resolve_context.table, field
        )
        if self.populate_field_object:
            self.field_object = output_field
        return ResolveResult(term=term, joins=joins, output_field=output_field)

    def _resolve_argument(
        self, resolve_context: ResolveContext, value: Any, *, treat_str_as_field: bool
    ) -> ResolveResult:
        if isinstance(value, Expression):
            return value.resolve(resolve_context)
        if isinstance(value, Term):
            return ResolveResult(term=value)
        if isinstance(value, str) and treat_str_as_field:
            return self._resolve_nested_field(resolve_context, value)
        return ResolveResult(term=value)

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        """
        Used to resolve the Function statement for SQL generation.

        :param model: Model the function is applied on to.
        :param table: ``pypika_tortoise.Table`` to keep track of the virtual SQL table
            (to allow self referential joins)
        :return: Dict with keys ``"joins"`` and ``"fields"``
        """

        function_arg = self._resolve_argument(resolve_context, self.field, treat_str_as_field=True)
        resolved_defaults = [
            self._resolve_argument(resolve_context, value, treat_str_as_field=False)
            for value in self.default_values
        ]
        default_terms = [value.term for value in resolved_defaults]
        term = self._get_function_field(function_arg.term, *default_terms)
        joins = function_arg.joins
        for value in resolved_defaults:
            if value.joins:
                joins = list(set(joins + value.joins))
        res = ResolveResult(
            term=term,
            joins=joins,
            output_field=function_arg.output_field,  # type:ignore[call-overload]
        )

        if self.populate_field_object and (
            res_output_field := res.output_field  # type:ignore[call-overload]
        ):
            self.field_object = res_output_field

        return res

    def over(self, window: Window | None = None, **window_kwargs: Any) -> WindowFunction:
        """
        Turn this function/aggregate into a window function evaluated over ``window``.

        :param window: A :class:`Window` instance. If not given, one is built from
            ``window_kwargs`` (``partition_by``, ``order_by`` and ``rows``).

        .. code-block:: python3

            Sum("score").over(Window(partition_by="group", order_by="id"))
            Sum("score").over(partition_by="group", order_by="id")
        """
        return WindowFunction(self, window or Window(**window_kwargs))


def _resolve_window_frame_edge(value: int | None, is_start: bool) -> Any:
    if value is None:
        return Preceding() if is_start else Following()
    if value == 0:
        return CURRENT_ROW
    if value < 0:
        return Preceding(-value)
    return Following(value)


class Window:
    """
    Window specification for window functions: optional ``PARTITION BY``, ``ORDER BY``
    and a ``ROWS`` frame.

    :param partition_by: Field name(s) (or ``F``/``Expression``) to partition by.
    :param order_by: Field name(s) to order the window by. Prefix with ``-`` for
        descending order, e.g. ``"-score"``.
    :param rows: Optional ``(start, end)`` tuple describing a ``ROWS BETWEEN <start>
        AND <end>`` frame. Each bound is ``None`` for ``UNBOUNDED PRECEDING``/``UNBOUNDED
        FOLLOWING``, ``0`` for ``CURRENT ROW``, a negative int ``-n`` for ``n PRECEDING``
        or a positive int ``n`` for ``n FOLLOWING``.

    .. code-block:: python3

        Window(partition_by="group", order_by="-score")
        Window(order_by="id", rows=(None, 0))  # ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    """

    __slots__ = ("partition_by", "order_by", "rows")

    def __init__(
        self,
        partition_by: str
        | F
        | Expression
        | Term
        | Iterable[str | F | Expression | Term]
        | None = None,
        order_by: str | F | Expression | Iterable[str | F | Expression] | None = None,
        rows: tuple[int | None, int | None] | None = None,
    ) -> None:
        if partition_by is None:
            self.partition_by: list[Any] = []
        elif isinstance(partition_by, (str, Expression, Term)):
            self.partition_by = [partition_by]
        else:
            self.partition_by = list(partition_by)
        if order_by is None:
            self.order_by: list[Any] = []
        elif isinstance(order_by, (str, Expression)):
            self.order_by = [order_by]
        else:
            self.order_by = list(order_by)
        self.rows = None
        if rows is not None:
            self.rows = self._validate_rows(rows)

    @staticmethod
    def _validate_rows(rows: tuple[int | None, int | None]) -> tuple[int | None, int | None]:
        try:
            start, end = rows
        except (TypeError, ValueError):
            raise OperationalError(
                "Window rows frame must be a (start, end) tuple, "
                "e.g. rows=(None, 0) for ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            ) from None
        for bound in (start, end):
            if bound is not None and not isinstance(bound, int):
                raise OperationalError(
                    f"Window rows frame bounds must be int or None, got {bound!r}"
                )
        # Position of a bound on the frame axis: UNBOUNDED PRECEDING < n PRECEDING
        # < CURRENT ROW < n FOLLOWING < UNBOUNDED FOLLOWING.
        start_pos = start if start is not None else -math.inf
        end_pos = end if end is not None else math.inf
        if start_pos > end_pos:
            raise OperationalError(
                f"Unsupported window rows frame ({start}, {end}): "
                "the start bound must not come after the end bound"
            )
        return (start, end)

    def _resolve_part(
        self, resolve_context: ResolveContext, part: Any
    ) -> ResolveResult:
        if isinstance(part, str):
            return F(part).resolve(resolve_context)
        if isinstance(part, Expression):
            return part.resolve(resolve_context)
        if isinstance(part, Term):
            return ResolveResult(term=part)
        raise OperationalError(f"Invalid window specification part {part!r}")

    def resolve(
        self, resolve_context: ResolveContext
    ) -> tuple[list[Term], list[tuple[Term, Order]], list[TableCriterionTuple]]:
        """
        Resolve the window specification into partition terms, ordering terms and the
        joins they require.
        """
        joins: list[TableCriterionTuple] = []
        partition_terms: list[Term] = []
        for part in self.partition_by:
            result = self._resolve_part(resolve_context, part)
            partition_terms.append(result.term)
            joins.extend(result.joins)

        order_terms: list[tuple[Term, Order]] = []
        for item in self.order_by:
            order = Order.asc
            if isinstance(item, str) and item.startswith("-"):
                order = Order.desc
                item = item[1:]
            result = self._resolve_part(resolve_context, item)
            order_terms.append((result.term, order))
            joins.extend(result.joins)
        return partition_terms, order_terms, joins


class WindowFunction(Expression):
    """
    A function evaluated over a :class:`Window` (SQL ``OVER (...)`` clause).

    Either wraps an existing :class:`Function`/:class:`Aggregate` (e.g. ``Sum``) or is
    subclassed for dedicated window functions such as ``RowNumber``.

    :param func: The function/aggregate to wrap. Not used by subclasses that define
        their own ``database_func``.
    :param window: The :class:`Window` specification. Defaults to an empty ``OVER ()``.

    .. attribute:: database_func
        :annotation: pypika_tortoise.terms.AnalyticFunction | None

        The pypika analytic function this represents, if it is a dedicated window
        function. ``None`` when wrapping a regular function/aggregate.
    """

    __slots__ = ("func", "window", "field_object")

    database_func: type[AnalyticFunction] | None = None

    def __init__(self, func: Function | None = None, window: Window | None = None) -> None:
        if func is None and self.database_func is None:
            raise OperationalError(
                "WindowFunction requires a function/aggregate to wrap, "
                "e.g. WindowFunction(Sum('score'), Window(...)) or Sum('score').over(...)"
            )
        self.func = func
        self.window = window or Window()
        self.field_object: Field | None = None

    def _resolve_function_term(
        self, resolve_context: ResolveContext
    ) -> tuple[AnalyticFunction, list[TableCriterionTuple], Field | None]:
        if self.database_func is not None:
            return self.database_func(), [], None

        func = cast(Function, self.func)
        func_result = func.resolve(resolve_context)
        pypika_func = func_result.term
        name = getattr(pypika_func, "name", None)
        if name is None or isinstance(pypika_func, AnalyticFunction):
            raise OperationalError(
                f"{type(func).__name__} can not be used as a window function"
            )
        if getattr(pypika_func, "_distinct", False):
            raise OperationalError("DISTINCT is not supported in window functions")
        analytic = WindowFrameAnalyticFunction(name, *pypika_func.args)
        return analytic, func_result.joins, func_result.output_field

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        capabilities = resolve_context.model._meta.db.capabilities
        if not capabilities.support_window_functions:
            raise UnSupportedError(
                f"Window functions are not supported by dialect {capabilities.dialect!r}"
            )
        if self.window.rows is not None and not capabilities.support_window_frame_rows:
            raise UnSupportedError(
                f"ROWS window frames are not supported by dialect {capabilities.dialect!r}"
            )

        analytic, func_joins, output_field = self._resolve_function_term(resolve_context)
        partition_terms, order_terms, window_joins = self.window.resolve(resolve_context)

        analytic = analytic.over(*partition_terms)
        for term, order in order_terms:
            analytic = analytic.orderby(term, order=order)
        if self.window.rows is not None:
            start, end = self.window.rows
            analytic = analytic.rows(
                _resolve_window_frame_edge(start, is_start=True),
                _resolve_window_frame_edge(end, is_start=False),
            )

        if self.func is not None and getattr(self.func, "populate_field_object", False):
            self.field_object = getattr(self.func, "field_object", None)

        joins = list(set(func_joins + window_joins))
        return ResolveResult(term=analytic, joins=joins, output_field=output_field)


class Aggregate(Function):
    """
    Base for SQL Aggregates.

    :param field: Field name
    :param default_values: Extra parameters to the function.
    :param distinct: Flag for aggregate with distinction
    """

    database_func: type[AggregateFunction] = DistinctOptionFunction

    def __init__(
        self,
        field: str | F | CombinedExpression,
        *default_values: Any,
        distinct: bool = False,
        _filter: Q | None = None,
    ) -> None:
        super().__init__(field, *default_values)
        self.distinct = distinct
        self.filter = _filter

    def _get_function_field(  # type:ignore[override]
        self, field: ArithmeticExpression | PypikaField | str, *default_values
    ) -> DistinctOptionFunction:
        function = cast(DistinctOptionFunction, self.database_func(field, *default_values))
        if self.distinct:
            function = function.distinct()
        return function

    def _resolve_nested_field(self, resolve_context: ResolveContext, field: str) -> ResolveResult:
        ret = super()._resolve_nested_field(resolve_context, field)
        if self.filter:
            modifier = QueryModifier()
            modifier &= self.filter.resolve(resolve_context)
            ret.term = PypikaCase().when(modifier.where_criterion, ret.term).else_(None)

        return ret


class _WhenThen(Term):
    """This is not a real term, but a helper to store the when and then terms."""

    def __init__(self, when: Term, then: Term) -> None:
        self.when = when
        self.then = then


class When(Expression):
    """
    When expression.

    :param args: Q objects
    :param kwargs: keyword criterion like filter
    :param then: value for criterion
    :param negate: false (default)
    """

    def __init__(
        self,
        *args: Q,
        then: str | F | CombinedExpression | Function,
        negate: bool = False,
        **kwargs: Any,
    ) -> None:
        self.args = args
        self.then = then
        self.negate = negate
        self.kwargs = kwargs

    def _resolve_q_objects(self) -> list[Q]:
        q_objects = []
        for arg in self.args:
            if not isinstance(arg, Q):
                raise TypeError("expected Q objects as args")
            if self.negate:
                q_objects.append(~arg)
            else:
                q_objects.append(arg)

        for key, value in self.kwargs.items():
            if self.negate:
                q_objects.append(~Q(**{key: value}))
            else:
                q_objects.append(Q(**{key: value}))
        return q_objects

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        q_objects = self._resolve_q_objects()

        modifier = QueryModifier()
        for node in q_objects:
            modifier &= node.resolve(resolve_context)

        if isinstance(self.then, Expression):
            then = self.then.resolve(resolve_context).term
        else:
            then = cast(Term, Term.wrap_constant(self.then))

        return ResolveResult(term=_WhenThen(modifier.where_criterion, then))


class Case(Expression):
    """
    Case expression.

    :param args: When objects
    :param default: value for 'CASE WHEN ... THEN ... ELSE <default> END'
    """

    def __init__(
        self,
        *args: When,
        default: str | F | CombinedExpression | Function | None = None,
    ) -> None:
        self.args = args
        self.default = default

    def resolve(self, resolve_context: ResolveContext) -> ResolveResult:
        case = PypikaCase()
        for arg in self.args:
            if not isinstance(arg, When):
                raise TypeError("expected When objects as args")
            when = arg.resolve(resolve_context)
            when_term = cast(_WhenThen, when.term)
            case = case.when(when_term.when, when_term.then)

        if isinstance(self.default, Expression):
            case = case.else_(self.default.resolve(resolve_context).term)
        else:
            case = case.else_(Term.wrap_constant(self.default))

        return ResolveResult(term=case)
