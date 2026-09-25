"""
Tests for window (analytic) function support.

Uses real window functions on SQLite (the default TORTOISE_TEST_DB).
"""

import os

import pytest
import pytest_asyncio

from tortoise import fields
from tortoise.context import tortoise_test_context
from tortoise.exceptions import OperationalError
from tortoise.expressions import Q, RowNumber, Rows
from tortoise.functions import Avg, Count, Sum
from tortoise.models import Model


class WindowTeam(Model):
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=50)


class WindowScore(Model):
    id = fields.IntField(primary_key=True)
    player = fields.CharField(max_length=50)
    category = fields.CharField(max_length=50)
    points = fields.IntField(null=True)
    team: fields.ForeignKeyNullableRelation[WindowTeam] = fields.ForeignKeyField(
        "models.WindowTeam", related_name="scores", null=True
    )


@pytest_asyncio.fixture(scope="function")
async def db_window():
    db_url = os.getenv("TORTOISE_TEST_DB", "sqlite://:memory:")
    async with tortoise_test_context(
        modules=["tests.test_window_functions"],
        db_url=db_url,
        app_label="models",
        connection_label="models",
    ) as ctx:
        yield ctx


@pytest_asyncio.fixture(scope="function")
async def scores(db_window):
    data = [
        # player, category, points
        ("p1", "a", 10),
        ("p2", "a", 10),  # tie with p1
        ("p3", "a", None),  # NULL points
        ("p4", "b", 5),
        ("p5", "b", 7),
        ("p6", "b", 7),  # tie with p5
    ]
    for player, category, points in data:
        await WindowScore.create(player=player, category=category, points=points)
    return db_window


@pytest.mark.asyncio
async def test_row_number_per_partition(scores):
    """Leaderboard: row_number per group, keep top 2 of each group."""
    qs = (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__lte=2)
        .order_by("category", "rn")
    )
    sql = qs.sql()
    # the window function is computed in a subquery and filtered in the outer query
    assert 'ROW_NUMBER() OVER(PARTITION BY "category" ORDER BY "points" DESC)' in sql
    assert ') "_window" WHERE "rn"<=' in sql
    assert "OVER" not in sql.split(') "_window"')[1]

    result = await qs
    assert [(row.player, row.category, row.rn) for row in result] == [
        ("p1", "a", 1),
        ("p2", "a", 2),
        ("p5", "b", 1),
        ("p6", "b", 2),
    ]


@pytest.mark.asyncio
async def test_row_number_ties(scores):
    """Tied values still get distinct row numbers."""
    result = await WindowScore.annotate(
        rn=RowNumber(partition_by="category", order_by="-points")
    ).filter(category="a")
    ranks = sorted(row.rn for row in result)
    assert ranks == [1, 2, 3]
    # the two tied players share the top ranks
    assert {row.player for row in result if row.rn <= 2} == {"p1", "p2"}


@pytest.mark.asyncio
async def test_row_number_nulls(scores):
    """NULL values sort last on descending order and get the last row number."""
    result = await WindowScore.annotate(
        rn=RowNumber(partition_by="category", order_by="-points")
    ).filter(category="a")
    by_player = {row.player: row.rn for row in result}
    assert by_player["p3"] == 3
    # filtering the NULL row out through the window alias works
    top = (
        await WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(category="a", rn__lte=2)
        .order_by("rn")
    )
    assert [row.player for row in top] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_empty_partition_result(scores):
    """No rows in a partition (or at all) means an empty result, not an error."""
    result = await WindowScore.annotate(
        rn=RowNumber(partition_by="category", order_by="-points")
    ).filter(category="does-not-exist", rn__lte=3)
    assert result == []

    await WindowScore.all().delete()
    result = await WindowScore.annotate(
        rn=RowNumber(partition_by="category", order_by="-points")
    ).filter(rn__lte=3)
    assert result == []


@pytest.mark.asyncio
async def test_rows_frame_running_total(scores):
    """ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW gives a running total."""
    result = await WindowScore.annotate(
        running=Sum("points").over(partition_by="category", order_by="id", rows=Rows(None, 0))
    ).order_by("id")
    assert [(row.player, row.running) for row in result] == [
        ("p1", 10),
        ("p2", 20),
        ("p3", 20),  # NULL points add nothing
        ("p4", 5),
        ("p5", 12),
        ("p6", 19),
    ]


@pytest.mark.asyncio
async def test_rows_frame_preceding_and_following(scores):
    """Frame boundaries before and after the current row."""
    result = await WindowScore.annotate(
        windowed=Sum("points").over(order_by="id", rows=Rows(-1, 1))
    ).order_by("id")
    # sum of previous, current and next row (points treated as 0 when NULL)
    assert [row.windowed for row in result] == [20, 20, 15, 12, 19, 14]

    result = await WindowScore.annotate(
        previous=Sum("points").over(order_by="id", rows=Rows(-1, -1))
    ).order_by("id")
    # sum of exactly the previous row; the first row has no predecessor -> NULL
    assert [row.previous for row in result] == [None, 10, 10, None, 5, 7]


@pytest.mark.asyncio
async def test_window_aggregate_avg(scores):
    """Existing aggregates can be wrapped as window functions."""
    result = await WindowScore.annotate(
        avg_points=Avg("points").over(partition_by="category")
    ).order_by("id")
    by_player = {row.player: row.avg_points for row in result}
    assert by_player["p1"] == 10
    assert by_player["p4"] == pytest.approx(19 / 3)


@pytest.mark.asyncio
async def test_invalid_frame_raises_before_execution(scores):
    # start after end
    with pytest.raises(OperationalError, match="Invalid ROWS frame"):
        await WindowScore.annotate(
            total=Sum("points").over(order_by="id", rows=Rows(1, -1))
        ).all()
    with pytest.raises(OperationalError, match="Invalid ROWS frame"):
        await WindowScore.annotate(
            total=Sum("points").over(order_by="id", rows=Rows(0, -1))
        ).all()
    # frames are not supported by ROW_NUMBER
    with pytest.raises(OperationalError, match="does not support window frames"):
        await WindowScore.annotate(
            rn=RowNumber(order_by="id").over(order_by="id", rows=Rows(None, 0))
        ).all()
    # DISTINCT is not supported for window functions on SQLite
    with pytest.raises(OperationalError, match="DISTINCT is not supported"):
        await WindowScore.annotate(
            total=Count("id", distinct=True).over(partition_by="category")
        ).all()


@pytest.mark.asyncio
async def test_window_filter_keeps_other_filters_ordering_pagination(scores):
    """Regular filters stay in the inner query; ordering and limit/offset apply
    to the final (outer) result."""
    qs = (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(points__isnull=False, rn__lte=3)
        .order_by("category", "rn")
        .offset(1)
        .limit(2)
    )
    sql = qs.sql()
    assert sql.index("LIMIT") > sql.index(') "_window"')
    result = await qs
    assert [(row.player, row.rn) for row in result] == [("p2", 2), ("p5", 1)]


@pytest.mark.asyncio
async def test_window_filter_mixed_q_object(scores):
    """A Q object mixing window and regular filters is resolved in the outer query."""
    result = await (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(Q(rn__lte=1) | Q(category="b"))
        .order_by("category", "rn")
    )
    assert [(row.player, row.category, row.rn) for row in result] == [
        ("p1", "a", 1),
        ("p5", "b", 1),
        ("p6", "b", 2),
        ("p4", "b", 3),
    ]


@pytest.mark.asyncio
async def test_multiple_windows(scores):
    """Several window annotations in one query, one of them filtered on."""
    result = await (
        WindowScore.annotate(
            rn=RowNumber(partition_by="category", order_by="-points"),
            running=Sum("points").over(
                partition_by="category", order_by="id", rows=Rows(None, 0)
            ),
        )
        .filter(rn__lte=2)
        .order_by("category", "rn")
    )
    assert [(row.player, row.rn, row.running) for row in result] == [
        ("p1", 1, 10),
        ("p2", 2, 20),
        ("p5", 1, 12),
        ("p6", 2, 19),
    ]


@pytest.mark.asyncio
async def test_window_count_and_exists(scores):
    qs = WindowScore.annotate(
        rn=RowNumber(partition_by="category", order_by="-points")
    ).filter(rn__lte=2)
    assert await qs.count() == 4
    assert await qs.exists() is True
    assert (
        await WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__gt=100)
        .exists()
    ) is False


@pytest.mark.asyncio
async def test_values_partial_fields(scores):
    """values()/values_list() only return the requested fields, even when the
    window annotation is only needed for filtering."""
    qs = (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__lte=2)
        .order_by("category", "rn")
        .values("player", "rn")
    )
    assert await qs == [
        {"player": "p1", "rn": 1},
        {"player": "p2", "rn": 2},
        {"player": "p5", "rn": 1},
        {"player": "p6", "rn": 2},
    ]

    # the window annotation is computed in the subquery but not returned
    result = await (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__lte=2)
        .order_by("category", "rn")
        .values("player")
    )
    assert result == [
        {"player": "p1"},
        {"player": "p2"},
        {"player": "p5"},
        {"player": "p6"},
    ]

    result = await (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__lte=2)
        .order_by("category", "rn")
        .values_list("player", "rn")
    )
    assert result == [("p1", 1), ("p2", 2), ("p5", 1), ("p6", 2)]

    # ordering by a field that is not selected still works
    result = await (
        WindowScore.annotate(rn=RowNumber(partition_by="category", order_by="-points"))
        .filter(rn__lte=1)
        .order_by("-points")
        .values("player")
    )
    assert result == [{"player": "p1"}, {"player": "p5"}]


@pytest.mark.asyncio
async def test_normal_aggregation_not_affected(scores):
    """Plain aggregate queries on the same model must not get a window subquery."""
    qs = WindowScore.annotate(total=Sum("points")).filter(total__gt=10)
    sql = qs.sql()
    assert "_window" not in sql
    assert "HAVING" in sql
    # aggregates keep their grouping semantics: each row is its own group
    result = await WindowScore.annotate(total=Sum("points")).filter(total__gte=10)
    assert sorted(row.player for row in result) == ["p1", "p2"]

    qs = WindowScore.annotate(c=Count("id"))
    assert "_window" not in qs.sql()


@pytest.mark.asyncio
async def test_unsupported_combinations_raise_before_execution(scores):
    annotation = RowNumber(partition_by="category", order_by="-points")
    with pytest.raises(OperationalError, match="update\\(\\)"):
        await WindowScore.annotate(rn=annotation).filter(rn__lte=1).update(points=0)
    with pytest.raises(OperationalError, match="delete\\(\\)"):
        await WindowScore.annotate(rn=annotation).filter(rn__lte=1).delete()
    with pytest.raises(OperationalError, match="select_related"):
        await WindowScore.annotate(rn=annotation).filter(rn__lte=1).select_related("team")
