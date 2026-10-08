from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from meetple_ai.contracts import (
    POLICY_CONTEXT_MAX_CHARS,
    ModerationAnalysisRequest,
    ModerationAnalysisResponse,
    ModerationDecision,
    PolicyCandidate,
    PolicyCandidates,
    PolicySearchPlan,
)
from meetple_ai.model import ModelOutputError


class ModerationModel(Protocol):
    async def prepare_moderation(self, request: ModerationAnalysisRequest) -> PolicySearchPlan: ...
    async def refine_moderation_search(
        self,
        request: ModerationAnalysisRequest,
        previous_plan: PolicySearchPlan,
    ) -> PolicySearchPlan: ...
    async def embed(self, semantic_query: str) -> list[float]: ...
    async def analyze_moderation(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        policies: list[PolicyCandidate],
    ) -> ModerationDecision: ...
    async def repair_moderation_decision(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        policies: list[PolicyCandidate],
        previous_decision: ModerationDecision,
        validation_error: str,
    ) -> ModerationDecision: ...


class PolicyTools(Protocol):
    async def search_policies(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        query_embedding: list[float],
        limit: int,
    ) -> PolicyCandidates: ...


class ModerationState(TypedDict, total=False):
    request: ModerationAnalysisRequest
    input_validated: bool
    plan: PolicySearchPlan
    query_embedding: list[float]
    policies: PolicyCandidates
    policy_search_rewrites: int
    decision: ModerationDecision
    decision_repairs: int
    decision_validated: bool
    grounding_error: str
    response: ModerationAnalysisResponse


ALLOWED_ACTIONS_BY_RISK = {
    "LOW": {"DISMISS", "WARNING", "MANUAL_REVIEW"},
    "MEDIUM": {"WARNING", "SUSPEND_1_DAY", "SUSPEND_3_DAYS", "MANUAL_REVIEW"},
    "HIGH": {
        "SUSPEND_3_DAYS",
        "SUSPEND_7_DAYS",
        "PERMANENT_SUSPENSION",
        "FORCE_DELETE_MEETING",
        "MANUAL_REVIEW",
    },
    "CRITICAL": {
        "SUSPEND_7_DAYS",
        "PERMANENT_SUSPENSION",
        "FORCE_DELETE_MEETING",
        "MANUAL_REVIEW",
    },
}


def normalize_recommended_action(decision: ModerationDecision, target_type: str) -> ModerationDecision:
    action = decision.recommendedAction
    allowed = action in ALLOWED_ACTIONS_BY_RISK[decision.riskLevel]
    target_matches = action != "FORCE_DELETE_MEETING" or target_type == "MEETING"
    if allowed and target_matches:
        return decision
    return decision.model_copy(update={"recommendedAction": "MANUAL_REVIEW"})


def merge_repaired_grounding(
    original: ModerationDecision,
    repaired: ModerationDecision,
) -> ModerationDecision:
    # 보정 모델은 근거만 고칠 수 있다. 분류·위험도·우선순위·사유를 바꿔 제재 수위를
    # 올리지 못하게 하고, 확신도 감소와 보수적인 수동 검토/기각 전환만 허용한다.
    recommended_action = (
        repaired.recommendedAction
        if repaired.recommendedAction in {"MANUAL_REVIEW", "DISMISS"}
        else original.recommendedAction
    )
    return original.model_copy(
        update={
            "evidence": repaired.evidence,
            "policies": repaired.policies,
            "confidence": min(original.confidence, repaired.confidence),
            "recommendedAction": recommended_action,
        }
    )


def filter_relevant_policies(
    policies: PolicyCandidates,
    *,
    min_hybrid_score: float,
    result_limit: int,
) -> PolicyCandidates:
    relevant = [policy for policy in policies.items if policy.hybridScore >= min_hybrid_score]
    return PolicyCandidates(
        items=relevant[:result_limit],
        hasMore=policies.hasMore or len(relevant) > result_limit,
    )


def _validate_grounding(state: ModerationState) -> ModerationDecision:
    request = state["request"]
    decision = state["decision"]
    evidence_by_id = {item.evidenceId: item for item in request.evidence}
    policies_by_chunk = {item.policyChunkId: item for item in state["policies"].items}

    evidence_ids: set[int] = set()
    for grounding in decision.evidence:
        evidence = evidence_by_id.get(grounding.evidenceId)
        if (
            evidence is None
            or grounding.evidenceId in evidence_ids
            or grounding.evidenceQuote not in evidence.content
        ):
            raise ModelOutputError("신고 분석의 증거 근거를 확인할 수 없습니다.")
        evidence_ids.add(grounding.evidenceId)

    policy_chunks: set[int] = set()
    for grounding in decision.policies:
        policy = policies_by_chunk.get(grounding.policyChunkId)
        if (
            policy is None
            or policy.policyId != grounding.policyId
            or grounding.policyChunkId in policy_chunks
            or grounding.policyQuote not in policy.content[:POLICY_CONTEXT_MAX_CHARS]
        ):
            raise ModelOutputError("신고 분석의 정책 근거를 확인할 수 없습니다.")
        policy_chunks.add(grounding.policyChunkId)
    return normalize_recommended_action(decision, request.targetType)


def build_moderation_graph(
    model: ModerationModel,
    tools: PolicyTools,
    *,
    min_policy_hybrid_score: float = 0.30,
    policy_result_limit: int = 5,
):
    def validate_input(state: ModerationState):
        return {"input_validated": True}

    async def summarize_report(state: ModerationState):
        return {"plan": await model.prepare_moderation(state["request"])}

    async def embed_policy_query(state: ModerationState):
        return {"query_embedding": await model.embed(state["plan"].semanticQuery)}

    async def retrieve_policies(state: ModerationState):
        policies = await tools.search_policies(
            state["request"],
            state["plan"],
            state["query_embedding"],
            policy_result_limit,
        )
        return {
            "policies": filter_relevant_policies(
                policies,
                min_hybrid_score=min_policy_hybrid_score,
                result_limit=policy_result_limit,
            )
        }

    async def rewrite_policy_query(state: ModerationState):
        refined_plan = await model.refine_moderation_search(state["request"], state["plan"])
        return {
            # 재검색은 검색어만 넓히며 최초 신고 요약은 모델이 다시 쓰지 못하게 고정한다.
            "plan": refined_plan.model_copy(update={"summary": state["plan"].summary}),
            "policy_search_rewrites": state.get("policy_search_rewrites", 0) + 1,
        }

    def fail_policy_search(state: ModerationState):
        raise ModelOutputError("신고 분석에 적용할 운영 정책을 찾지 못했습니다.")

    async def classify_report(state: ModerationState):
        return {
            "decision": await model.analyze_moderation(
                state["request"], state["plan"], state["policies"].items
            )
        }

    def validate_evidence_and_policy(state: ModerationState):
        try:
            return {
                "decision": _validate_grounding(state),
                "decision_validated": True,
                "grounding_error": "",
            }
        except ModelOutputError as exc:
            return {
                "decision_validated": False,
                "grounding_error": str(exc),
            }

    async def repair_decision(state: ModerationState):
        repaired = await model.repair_moderation_decision(
            state["request"],
            state["plan"],
            state["policies"].items,
            state["decision"],
            state["grounding_error"],
        )
        return {
            "decision": merge_repaired_grounding(state["decision"], repaired),
            "decision_repairs": state.get("decision_repairs", 0) + 1,
        }

    def fail_grounding(state: ModerationState):
        raise ModelOutputError(state["grounding_error"])

    def build_result(state: ModerationState):
        decision = state["decision"]
        return {
            "response": ModerationAnalysisResponse(
                reportId=state["request"].reportId,
                reportType=decision.reportType,
                riskLevel=decision.riskLevel,
                priority=decision.priority,
                summary=state["plan"].summary,
                rationale=decision.rationale,
                evidenceIds=[item.evidenceId for item in decision.evidence],
                policyIds=list(dict.fromkeys(item.policyId for item in decision.policies)),
                confidence=decision.confidence,
                recommendedAction=decision.recommendedAction,
            )
        }

    graph = StateGraph(ModerationState)
    nodes = [
        ("validate_input", validate_input),
        ("summarize_report", summarize_report),
        ("embed_policy_query", embed_policy_query),
        ("retrieve_policies", retrieve_policies),
        ("rewrite_policy_query", rewrite_policy_query),
        ("fail_policy_search", fail_policy_search),
        ("classify_report", classify_report),
        ("validate_evidence_and_policy", validate_evidence_and_policy),
        ("repair_decision", repair_decision),
        ("fail_grounding", fail_grounding),
        ("build_result", build_result),
    ]
    for name, node in nodes:
        graph.add_node(name, node)
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "summarize_report")
    graph.add_edge("summarize_report", "embed_policy_query")
    graph.add_edge("embed_policy_query", "retrieve_policies")
    graph.add_conditional_edges(
        "retrieve_policies",
        lambda state: (
            "classify_report"
            if state["policies"].items
            else "rewrite_policy_query"
            if state.get("policy_search_rewrites", 0) < 1
            else "fail_policy_search"
        ),
        {
            "classify_report": "classify_report",
            "rewrite_policy_query": "rewrite_policy_query",
            "fail_policy_search": "fail_policy_search",
        },
    )
    graph.add_edge("rewrite_policy_query", "embed_policy_query")
    graph.add_edge("classify_report", "validate_evidence_and_policy")
    graph.add_conditional_edges(
        "validate_evidence_and_policy",
        lambda state: (
            "build_result"
            if state["decision_validated"]
            else "repair_decision"
            if state.get("decision_repairs", 0) < 1
            else "fail_grounding"
        ),
        {
            "build_result": "build_result",
            "repair_decision": "repair_decision",
            "fail_grounding": "fail_grounding",
        },
    )
    graph.add_edge("repair_decision", "validate_evidence_and_policy")
    graph.add_edge("build_result", END)
    return graph.compile()
