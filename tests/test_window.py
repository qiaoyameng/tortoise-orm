"""Tests for window function expressions (ROW_NUMBER and windowed aggregates)."""

import pytest
import pytest_asyncio

from tests.testmodels import Score, Tournament
from tortoise.contrib import test
from tortoise.exceptions import FieldError, OperationalError, UnSupportedError
from tortoise.expressions import Q, Window, WindowFunction
from tortoise.functions import Avg, Count, Max, Min, RowNumber, Sum

SCORES = [
    # (category, player, score)
    ("math", "amy", 90),
    ("math", "bob", 80),
    ("math", "cy", 80),  # tie with bob
    ("math", "dan", 70),
    ("chem", "eli", 95),
    ("chem", "fin", 85),
    ("chem", "gus", None),  # NULL score
    (None, "hal", 60),  # NULL category
]


@pytest_asyncio.fixture
async def scores(db):
    for category, player, score in SCORES:
        await Score.create(category=category, player=player, score=score)


def ranked(rows):
    return {row.player: row.rn for row in rows}


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_row_number_partition_and_order(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).order_by("category", "rn")
    assert ranked(rows) == {
        "amy": 1,
        "bob": 2,
        "cy": 3,
        "dan": 4,
        "eli": 1,
        "fin": 2,
        "gus": 3,
        "hal": 1,
    }


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_row_number_ties(db, scores):
    # Tied scores get distinct row numbers; with a deterministic secondary
    # ordering the lower id comes first.
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(category="math")
    assert ranked(rows)["bob"] == 2
    assert ranked(rows)["cy"] == 3

    # Without a secondary ordering the two tied rows still get distinct numbers
    rows = await Score.annotate(rn=RowNumber(partition_by="category", order_by="-score")).filter(
        category="math"
    )
    tie_ranks = sorted(rn for player, rn in ranked(rows).items() if player in ("bob", "cy"))
    assert tie_ranks == [2, 3]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_row_number_null_values(db, scores):
    # NULL category rows form a partition of their own
    rows = await Score.annotate(rn=RowNumber(partition_by="category", order_by="-score")).filter(
        category__isnull=True
    )
    assert ranked(rows) == {"hal": 1}

    # NULL scores sort last in descending order on SQLite/Postgres
    rows = await Score.annotate(rn=RowNumber(partition_by="category", order_by="-score")).filter(
        category="chem"
    )
    assert ranked(rows)["gus"] == 3


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_row_number_empty_result(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by="-score")
    ).filter(category="no-such-category")
    assert rows == []


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_row_number_top_n_per_group(db, scores):
    # The leaderboard case: rank within each group, then keep the top 2 per group
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=2)
    assert ranked(rows) == {"amy": 1, "bob": 2, "eli": 1, "fin": 2, "hal": 1}


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_exclude(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).exclude(rn=1)
    assert "amy" not in {row.player for row in rows}
    assert ranked(rows)["bob"] == 2


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_uses_outer_query(db, scores):
    qs = Score.annotate(rn=RowNumber(partition_by="category", order_by="-score")).filter(
        category="math", rn__lte=2
    )
    sql = qs.sql()
    # the window function is computed in the inner query...
    assert "ROW_NUMBER() OVER(" in sql
    inner, __, outer = sql.partition(') "_window" ')
    assert "ROW_NUMBER() OVER(" in inner
    # ...while the filter on the window alias is applied by the outer query
    assert '"_window"."rn"<=' in sql
    assert '"_window"."rn"<=' in outer
    # the regular filter stays in the inner query
    assert '"category"=' in inner
    assert [row.player for row in await qs] == ["amy", "bob"]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_keeps_ordering_and_pagination(db, scores):
    qs = (
        Score.annotate(rn=RowNumber(partition_by="category", order_by=["-score", "id"]))
        .filter(rn__lte=2)
        .order_by("-score")
        .limit(2)
        .offset(1)
    )
    sql = qs.sql()
    # ordering and pagination apply on the outer query, after the window filter
    outer = sql.split(') "_window" ', 1)[1]
    assert outer.index("ORDER BY") > outer.index('"_window"."rn"<=')
    assert outer.index("LIMIT") > outer.index("ORDER BY")
    rows = await qs
    assert [(row.player, row.rn) for row in rows] == [("amy", 1), ("fin", 2)]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_with_q_objects(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(Q(category="math") & Q(rn__lte=1))
    assert [row.player for row in rows] == ["amy"]

    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(Q(rn=1) | Q(rn=2), category="chem")
    assert sorted(row.player for row in rows) == ["eli", "fin"]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_related_field_raises(db, scores):
    tournament = await Tournament.create(name="T1")
    await tournament.events.create(name="e1")
    with pytest.raises(FieldError, match="related fields"):
        await tournament.events.all().annotate(
            rn=RowNumber(partition_by="tournament_id", order_by="-event_id")
        ).filter(Q(rn=1) & Q(tournament__name="T1"))


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_unsupported_query_types(db, scores):
    with pytest.raises(OperationalError, match="window function"):
        await Score.annotate(rn=RowNumber(order_by="id")).filter(rn__lte=1).count()
    with pytest.raises(OperationalError, match="window function"):
        await Score.annotate(rn=RowNumber(order_by="id")).filter(rn__lte=1).exists()


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_sum_running_total(db, scores):
    rows = await Score.annotate(
        running=Sum("score").over(Window(order_by="id", rows=(None, 0)))
    ).order_by("id")
    expected = {}
    total = 0
    for row in rows:
        if row.score is not None:
            total += row.score
        expected[row.player] = total
    assert {row.player: row.running for row in rows} == expected


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_frame_preceding_and_following(db):
    for value in [1, 2, 3, 4, 5]:
        await Score.create(category="n", player=f"p{value}", score=value)
    rows = await Score.annotate(
        windowed=Sum("score").over(Window(order_by="score", rows=(-1, 1)))
    ).order_by("score")
    # frame covers the previous, current and next row, clipped at the boundaries
    assert [row.windowed for row in rows] == [3, 6, 9, 12, 9]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_frame_current_row_to_unbounded_following(db):
    for value in [1, 2, 3]:
        await Score.create(category="n", player=f"p{value}", score=value)
    rows = await Score.annotate(
        remaining=Sum("score").over(Window(order_by="score", rows=(0, None)))
    ).order_by("score")
    assert [row.remaining for row in rows] == [6, 5, 3]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_aggregates(db, scores):
    rows = await Score.annotate(
        total=Sum("score").over(partition_by="category"),
        cnt=Count("id").over(partition_by="category"),
        best=Max("score").over(partition_by="category"),
        worst=Min("score").over(partition_by="category"),
        avg=Avg("score").over(partition_by="category"),
    )
    row = next(row for row in rows if row.player == "amy")
    assert row.total == 320
    assert row.cnt == 4
    assert row.best == 90
    assert row.worst == 70
    assert row.avg == 80


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_multiple_windows_in_one_query(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"]),
        running=Sum("score").over(Window(partition_by="category", order_by="id", rows=(None, 0))),
    ).filter(rn__lte=2, category="math")
    assert {row.player: (row.rn, row.running) for row in rows} == {
        "amy": (1, 90),
        "bob": (2, 170),
    }


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_function_class_form(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(window=Window(partition_by="category", order_by=["-score", "id"])),
        total=WindowFunction(Sum("score"), Window(partition_by="category")),
    ).filter(rn=1, category="math")
    assert [(row.player, row.total) for row in rows] == [("amy", 320)]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_values_partial_fields(db, scores):
    # only a subset of fields is selected; the window alias is still filterable
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1).values("player", "rn")
    assert sorted(rows, key=lambda row: row["player"]) == [
        {"player": "amy", "rn": 1},
        {"player": "eli", "rn": 1},
        {"player": "hal", "rn": 1},
    ]

    # the window annotation itself does not have to be selected
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1).values("player")
    assert sorted(row["player"] for row in rows) == ["amy", "eli", "hal"]

    # a regular filter mixed with the window filter forces the column into the subquery
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(Q(category="math") & Q(rn__lte=1)).values("player")
    assert [row["player"] for row in rows] == ["amy"]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_values_list_partial_fields(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1).values_list("player", "rn")
    assert sorted(rows) == [("amy", 1), ("eli", 1), ("hal", 1)]

    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1).values_list("player", flat=True)
    assert sorted(rows) == ["amy", "eli", "hal"]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_values_ordering(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=2).order_by("-rn").values("player", "rn")
    assert [row["rn"] for row in rows] == [2, 2, 1, 1, 1]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_only(db, scores):
    rows = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1).only("id", "player")
    assert sorted(row.player for row in rows) == ["amy", "eli", "hal"]


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_filter_first(db, scores):
    row = await Score.annotate(
        rn=RowNumber(partition_by="category", order_by=["-score", "id"])
    ).filter(rn__lte=1, category="math").first()
    assert row.player == "amy"


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_normal_aggregation_unaffected(db, scores):
    rows = await Score.annotate(n=Count("id")).filter(n__gte=1)
    assert len(rows) == len(SCORES)

    rows = await Score.all().values("category")
    assert len(rows) == len(SCORES)

    # plain aggregation with group by still works on the same model
    rows = await Score.annotate(total=Sum("score")).group_by("category").values(
        "category", "total"
    )
    by_category = {row["category"]: row["total"] for row in rows}
    assert by_category == {"math": 320, "chem": 180, None: 60}


@pytest.mark.asyncio
async def test_window_functions_unsupported(db, scores):
    capabilities = Score._meta.db.capabilities
    capabilities.__dict__["support_window_functions"] = False
    try:
        with pytest.raises(UnSupportedError, match="Window functions are not supported"):
            await Score.annotate(rn=RowNumber(order_by="id")).all()
    finally:
        capabilities.__dict__["support_window_functions"] = True


@test.requireCapability(support_window_functions=True)
@pytest.mark.asyncio
async def test_window_frame_rows_unsupported(db, scores):
    capabilities = Score._meta.db.capabilities
    capabilities.__dict__["support_window_frame_rows"] = False
    try:
        with pytest.raises(UnSupportedError, match="ROWS window frames are not supported"):
            await Score.annotate(
                total=Sum("score").over(Window(order_by="id", rows=(None, 0)))
            ).all()
        # window functions without a frame still work
        rows = await Score.annotate(rn=RowNumber(order_by="id")).filter(player="amy")
        assert rows[0].rn == 1
    finally:
        capabilities.__dict__["support_window_frame_rows"] = True


@pytest.mark.asyncio
async def test_window_invalid_frames(db):
    with pytest.raises(OperationalError, match="start bound must not come after"):
        Window(order_by="id", rows=(1, -1))
    with pytest.raises(OperationalError, match="start bound must not come after"):
        Window(order_by="id", rows=(0, -1))
    with pytest.raises(OperationalError, match="must be int or None"):
        Window(order_by="id", rows=("x", 0))
    with pytest.raises(OperationalError, match="must be a \\(start, end\\) tuple"):
        Window(order_by="id", rows=(1,))


@pytest.mark.asyncio
async def test_window_distinct_unsupported(db):
    with pytest.raises(OperationalError, match="DISTINCT is not supported"):
        await Score.annotate(cnt=Count("id", distinct=True).over(order_by="id")).all()
