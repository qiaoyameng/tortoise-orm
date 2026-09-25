.. _functions:

======================
Functions & Aggregates
======================

To apply functions to values and get aggregates computed on the DB side, one needs to annotate the QuerySet.

.. code-block:: py3

    results = await SomeModel.filter(...).annotate(clean_desc=Coalesce("desc", "N/A"))

This will add a new attribute on each ``SomeModel`` instance called ``clean_desc`` that will now contain the annotated data.

One can also call ``.values()`` or ``.values_list()`` on it to get the data as per regular.

Functions
=========

Functions apply a transform on each instance of a Field.

.. autoclass:: tortoise.functions.Trim

.. autoclass:: tortoise.functions.Length

.. autoclass:: tortoise.functions.Coalesce

.. autoclass:: tortoise.functions.Lower

.. autoclass:: tortoise.functions.Upper

.. autoclass:: tortoise.functions.Concat

.. autoclass:: tortoise.contrib.mysql.functions.Rand

.. autoclass:: tortoise.contrib.postgres.functions.Random

.. autoclass:: tortoise.contrib.sqlite.functions.Random

Aggregates
==========

Aggregated apply on the entire column, and will often be used with grouping.
So often makes sense with a ``.first()`` QuerySet.

.. autoclass:: tortoise.functions.Count

.. autoclass:: tortoise.functions.Sum

.. autoclass:: tortoise.functions.Max

.. autoclass:: tortoise.functions.Min

.. autoclass:: tortoise.functions.Avg


Window functions
================

Window functions compute a value over a window of rows related to the current row,
using the SQL ``OVER (...)`` clause. A window is described by a
:class:`tortoise.expressions.Window`, which supports ``PARTITION BY``, ``ORDER BY``
and a ``ROWS`` frame:

.. code-block:: py3

    from tortoise.expressions import Window
    from tortoise.functions import RowNumber, Sum

    # rank scores within each category
    await Score.annotate(
        rank=RowNumber(partition_by="category", order_by="-score")
    )

    # running total per category
    await Score.annotate(
        running=Sum("score").over(
            Window(partition_by="category", order_by="id", rows=(None, 0))
        )
    )

Any existing aggregate (e.g. ``Sum``, ``Count``, ``Avg``, ``Min``, ``Max``) can be
turned into a window function with ``.over(...)``.

Window function results can not be filtered in the same query level, so when a
``filter()`` references a window annotation, Tortoise moves that filter to an
outer query automatically, keeping the other filters, ordering and pagination
semantics:

.. code-block:: py3

    # top 3 scores per category
    await Score.annotate(
        rank=RowNumber(partition_by="category", order_by="-score")
    ).filter(rank__lte=3)

Whether window functions and ``ROWS`` frames are supported depends on the database
(see ``support_window_functions`` and ``support_window_frame_rows`` capabilities);
unsupported combinations raise an error before the query is executed.

.. autoclass:: tortoise.functions.RowNumber

.. autoclass:: tortoise.expressions.Window

.. autoclass:: tortoise.expressions.WindowFunction


Base function class
===================

.. automodule:: tortoise.functions
    :members: Aggregate

    .. autoclass:: Function
        :members:

Custom functions
================
You can define custom functions which are not builtin, such as ``TruncMonth`` and ``JsonExtract`` etc.

.. code-block:: python3

    from pypika_tortoise import CustomFunction
    from tortoise.expressions import F, Function

    class TruncMonth(Function):
        database_func = CustomFunction("DATE_FORMAT", ["name", "dt_format"])

    sql = Task.all().annotate(date=TruncMonth('created_at', '%Y-%m-%d')).values('date').sql()
    print(sql)
    # SELECT DATE_FORMAT(`created_at`,'%Y-%m-%d') `date` FROM `task`

And you can also use functions in update, the example is only suitable for MySQL and SQLite, but PostgreSQL is the same.

.. code-block:: python3

    from tortoise.expressions import F
    from tortoise.functions import Function
    from pypika_tortoise.terms import Function as PupikaFunction

    class JsonSet(Function):
        class PypikaJsonSet(PupikaFunction):
            def __init__(self, field: F, expression: str, value: Any):
                super().__init__("JSON_SET", field, expression, value)

        database_func = PypikaJsonSet

    json = await JSONFields.create(data_default={"a": 1})
    json.data_default = JsonSet(F("data_default"), "$.a", 2)
    await json.save()

    # or use queryset.update()
    sql = JSONFields.filter(pk=json.pk).update(data_default=JsonSet(F("data_default"), "$.a", 3)).sql()
    print(sql)
    # UPDATE jsonfields SET data_default=JSON_SET(`data_default`,'$.a',3) where id=1
