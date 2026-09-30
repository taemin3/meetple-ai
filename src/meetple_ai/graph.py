import re
from datetime import date, datetime, time, timedelta
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from meetple_ai.contracts import (
    Candidates,
    Filters,
    Intent,
    SearchRequest,
    SearchResponse,
    Selection,
)
from meetple_ai.model import ModelOutputError, SearchModel


class MeetingTools(Protocol):
    async def categories(self) -> list[str]: ...
    async def search(self, filters: Filters) -> Candidates: ...


class SearchState(TypedDict, total=False):
    request: SearchRequest
    categories: list[str]
    intent: Intent
    filters: Filters
    candidates: Candidates
    selection: Selection
    response: SearchResponse


TIME_PATTERN = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d")


def parse_time_bound(value: str | None) -> time | None:
    if value is None:
        return None
    if TIME_PATTERN.fullmatch(value) is None:
        raise ValueError("검색할 시간을 다시 확인해주세요.")
    return time.fromisoformat(value)


def terminal(status: str, message: str) -> SearchResponse:
    return SearchResponse(status=status, message=message, filters=None, recommendations=[])


def resolve_filters(request: SearchRequest, intent: Intent, categories: list[str]) -> Filters:
    if intent.category is not None and intent.category not in categories:
        raise ValueError("검색할 카테고리를 다시 확인해주세요.")
    radius = intent.radiusMeters if intent.radiusMeters is not None else request.radiusMeters
    if not 100 <= radius <= request.radiusMeters:
        raise ValueError("앱에서 검색 반경을 조정한 뒤 다시 검색해주세요.")
    today = request.referenceTime.date()
    start, end = today, today + timedelta(days=30)
    if intent.dateMode == "today":
        end = today + timedelta(days=1)
    elif intent.dateMode == "tomorrow":
        start, end = today + timedelta(days=1), today + timedelta(days=2)
    elif intent.dateMode in ("this_weekend", "next_weekend"):
        start = today - timedelta(days=today.weekday()) + timedelta(days=5)
        if intent.dateMode == "next_weekend":
            start += timedelta(days=7)
        end = start + timedelta(days=2)
    elif intent.dateMode == "range":
        try:
            start, end = date.fromisoformat(intent.startDate or ""), date.fromisoformat(intent.endDate or "")
        except ValueError as exc:
            raise ValueError("검색할 날짜를 구체적으로 입력해주세요.") from exc
        if start > end or (end - start).days > 365:
            raise ValueError("검색할 날짜 범위를 다시 확인해주세요.")
        end += timedelta(days=1)
    starts_at = max(datetime.combine(start, time.min), request.referenceTime)
    ends_before = datetime.combine(end, time.min)
    if starts_at >= ends_before:
        raise ValueError("앞으로 열리는 모임의 날짜를 입력해주세요.")
    has_explicit_time = intent.startTime is not None or intent.endTime is not None
    if intent.timeMode != "range" and has_explicit_time:
        raise ValueError("검색할 시간을 다시 확인해주세요.")
    starts_at_time, ends_before_time = None, None
    if intent.timeMode == "morning":
        starts_at_time, ends_before_time = time(6), time(12)
    elif intent.timeMode == "afternoon":
        starts_at_time, ends_before_time = time(12), time(18)
    elif intent.timeMode == "evening":
        starts_at_time = time(18)
    elif intent.timeMode == "range":
        starts_at_time = parse_time_bound(intent.startTime)
        ends_before_time = parse_time_bound(intent.endTime)
        if starts_at_time is None and ends_before_time is None:
            raise ValueError("검색할 시간을 다시 확인해주세요.")
    if starts_at_time is not None and ends_before_time is not None and starts_at_time == ends_before_time:
        raise ValueError("검색할 시간 범위를 다시 확인해주세요.")
    return Filters(
        keyword=intent.keyword,
        category=intent.category,
        startsAt=starts_at,
        endsBefore=ends_before,
        startsAtTime=starts_at_time,
        endsBeforeTime=ends_before_time,
        latitude=request.latitude,
        longitude=request.longitude,
        radiusMeters=radius,
    )


def build_graph(model: SearchModel, tools: MeetingTools):
    async def prepare(state: SearchState):
        if state["request"].latitude is None:
            return {"response": terminal("INPUT_REQUIRED", "앱에서 검색 기준 위치를 선택해주세요.")}
        return {"categories": await tools.categories()}

    async def interpret(state: SearchState):
        return {"intent": await model.interpret(state["request"], state["categories"])}

    def resolve(state: SearchState):
        intent = state["intent"]
        if intent.unsupportedReason:
            return {"response": terminal("UNSUPPORTED", intent.unsupportedReason[:500])}
        try:
            return {"filters": resolve_filters(state["request"], intent, state["categories"])}
        except (ValueError, OverflowError):
            return {
                "response": terminal("INPUT_REQUIRED", "검색할 날짜·시간·카테고리·반경을 다시 확인해주세요.")
            }

    async def retrieve(state: SearchState):
        candidates = await tools.search(state["filters"])
        if not candidates.items:
            return {
                "response": SearchResponse(
                    status="NO_RESULTS",
                    message="조건에 맞는 모집 중인 모임이 없습니다.",
                    filters=state["filters"],
                    recommendations=[],
                )
            }
        return {"candidates": candidates}

    async def select(state: SearchState):
        return {"selection": await model.select(state["request"], state["candidates"].items)}

    def verify(state: SearchState):
        candidates = {m.id: m for m in state["candidates"].items}
        seen: set[int] = set()
        for recommendation in state["selection"].recommendations:
            candidate = candidates.get(recommendation.meetingId)
            if (
                candidate is None
                or candidate.id in seen
                or not (
                    recommendation.evidenceQuote in candidate.title
                    or recommendation.evidenceQuote in candidate.description
                )
            ):
                raise ModelOutputError("추천 결과의 원문 근거를 확인할 수 없습니다.")
            seen.add(candidate.id)
        recommendations = state["selection"].recommendations
        message = (
            "검색 조건과 소개글을 확인한 모임입니다."
            if recommendations
            else "검색 후보에서 선호 조건의 근거를 찾지 못했습니다."
        )
        if state["candidates"].hasMore:
            message += " 가까운 모임 20개를 확인한 결과이며, 조건을 구체화하면 다른 후보를 찾을 수 있습니다."
        return {
            "response": SearchResponse(
                status="COMPLETED" if recommendations else "NO_RESULTS",
                message=message,
                filters=state["filters"],
                recommendations=recommendations,
            )
        }

    graph = StateGraph(SearchState)
    for name, node in [
        ("prepare", prepare),
        ("interpret", interpret),
        ("resolve", resolve),
        ("retrieve", retrieve),
        ("select", select),
        ("verify", verify),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda s: END if "response" in s else "interpret")
    graph.add_edge("interpret", "resolve")
    graph.add_conditional_edges("resolve", lambda s: END if "response" in s else "retrieve")
    graph.add_conditional_edges("retrieve", lambda s: END if "response" in s else "select")
    graph.add_edge("select", "verify")
    graph.add_edge("verify", END)
    # 이 단계는 요청 단위 읽기 워크플로. 영속 체크포인트와 자동 에이전트 루프는 후속 범위.
    return graph.compile()
