from datetime import datetime

import pytest
from conftest import FakeModel, FakeTools

from meetple_ai.contracts import Recommendation
from meetple_ai.graph import build_graph, resolve_filters
from meetple_ai.model import ModelOutputError


async def test_real_graph_calls_search_and_returns_grounded_result(request_data, intent, candidate):
    model = FakeModel(intent, [Recommendation(meetingId=10, evidenceQuote="처음 달리는 분 환영")])
    tools = FakeTools([candidate])
    result = await build_graph(model, tools).ainvoke({"request": request_data})
    assert result["response"].status == "COMPLETED"
    assert result["response"].filters.startsAt == datetime(2026, 10, 3)
    assert result["response"].filters.endsBefore == datetime(2026, 10, 5)
    assert model.calls == [
        "interpret",
        ("embed", "초보자가 참여할 수 있는 러닝 모임"),
        "select",
    ]
    assert len(tools.calls) == 2
    assert len(tools.calls[1][1]) == 1536


async def test_missing_location_does_not_call_model_or_search(request_data, intent):
    request_data.latitude, request_data.longitude = None, None
    model, tools = FakeModel(intent), FakeTools()
    result = await build_graph(model, tools).ainvoke({"request": request_data})
    assert result["response"].status == "INPUT_REQUIRED"
    assert model.calls == tools.calls == []


async def test_empty_search_skips_second_model_call(request_data, intent):
    model = FakeModel(intent)
    result = await build_graph(model, FakeTools()).ainvoke({"request": request_data})
    assert result["response"].status == "NO_RESULTS"
    assert model.calls == ["interpret", ("embed", "초보자가 참여할 수 있는 러닝 모임")]


async def test_generic_search_skips_embedding_call(request_data, intent):
    intent.semanticQuery = None
    model, tools = FakeModel(intent), FakeTools()
    result = await build_graph(model, tools).ainvoke({"request": request_data})
    assert result["response"].status == "NO_RESULTS"
    assert model.calls == ["interpret"]
    assert tools.calls[1][1] is None


@pytest.mark.parametrize("field,value", [("radiusMeters", 5000), ("category", "없는 카테고리")])
async def test_invalid_model_filters_do_not_reach_search(request_data, intent, field, value):
    intent = intent.model_copy(update={field: value})
    tools = FakeTools()
    result = await build_graph(FakeModel(intent), tools).ainvoke({"request": request_data})
    assert result["response"].status == "INPUT_REQUIRED"
    assert tools.calls == ["categories"]


@pytest.mark.parametrize(
    "picks",
    [
        [{"meetingId": 99, "evidenceQuote": "처음 달리는 분 환영"}],
        [{"meetingId": 10, "evidenceQuote": "참가비 무료"}],
        [{"meetingId": 10, "evidenceQuote": "러닝"}, {"meetingId": 10, "evidenceQuote": "러닝"}],
    ],
)
async def test_invalid_ids_quotes_and_duplicates_are_rejected(request_data, intent, candidate, picks):
    model = FakeModel(intent, [Recommendation(**p) for p in picks])
    with pytest.raises(ModelOutputError):
        await build_graph(model, FakeTools([candidate])).ainvoke({"request": request_data})


async def test_explicit_unsupported_condition_never_searches(request_data, intent):
    intent.unsupportedReason = "다른 지역 검색은 아직 지원하지 않습니다."
    tools = FakeTools()
    result = await build_graph(FakeModel(intent), tools).ainvoke({"request": request_data})
    assert result["response"].status == "UNSUPPORTED"
    assert result["response"].message == intent.unsupportedReason
    assert tools.calls == ["categories"]


def test_sunday_this_weekend_does_not_jump_to_next_week(request_data, intent):
    request_data.referenceTime = datetime(2026, 10, 4, 15)
    filters = resolve_filters(request_data, intent, ["운동"])
    assert filters.startsAt == request_data.referenceTime
    assert filters.endsBefore == datetime(2026, 10, 5)


def test_date_range_includes_end_day_and_rejects_past(request_data, intent):
    intent.dateMode, intent.startDate, intent.endDate = "range", "2026-10-01", "2026-10-03"
    assert resolve_filters(request_data, intent, ["운동"]).endsBefore == datetime(2026, 10, 4)
    intent.startDate, intent.endDate = "2020-01-01", "2020-01-02"
    with pytest.raises(ValueError):
        resolve_filters(request_data, intent, ["운동"])


@pytest.mark.parametrize(
    "mode,start,end,expected_start,expected_end",
    [
        ("morning", None, None, "06:00:00", "12:00:00"),
        ("afternoon", None, None, "12:00:00", "18:00:00"),
        ("evening", None, None, "18:00:00", None),
        ("range", "15:00", "16:00", "15:00:00", "16:00:00"),
    ],
)
def test_time_expression_becomes_deterministic_filter(
    request_data, intent, mode, start, end, expected_start, expected_end
):
    intent.timeMode, intent.startTime, intent.endTime = mode, start, end
    filters = resolve_filters(request_data, intent, ["운동"])
    assert (filters.startsAtTime.isoformat() if filters.startsAtTime else None) == expected_start
    assert (filters.endsBeforeTime.isoformat() if filters.endsBeforeTime else None) == expected_end


@pytest.mark.parametrize("mode", ["any", "morning", "afternoon", "evening"])
def test_non_range_time_mode_rejects_explicit_bounds(request_data, intent, mode):
    intent.timeMode, intent.startTime, intent.endTime = mode, "15:00", None
    with pytest.raises(ValueError):
        resolve_filters(request_data, intent, ["운동"])


@pytest.mark.parametrize("value", ["5:00", "15:00:00", "15:00Z", "15:00+09:00"])
def test_time_range_requires_exact_hh_mm(request_data, intent, value):
    intent.timeMode, intent.startTime, intent.endTime = "range", value, "16:00"
    with pytest.raises(ValueError):
        resolve_filters(request_data, intent, ["운동"])


def test_time_range_can_cross_midnight(request_data, intent):
    intent.timeMode, intent.startTime, intent.endTime = "range", "23:30", "00:30"
    filters = resolve_filters(request_data, intent, ["운동"])
    assert filters.startsAtTime == datetime.strptime("23:30", "%H:%M").time()
    assert filters.endsBeforeTime == datetime.strptime("00:30", "%H:%M").time()
