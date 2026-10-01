from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from meetple_ai.contracts import (
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
    async def embed(self, semantic_query: str) -> list[float]: ...
    async def analyze_moderation(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        policies: list[PolicyCandidate],
    ) -> ModerationDecision: ...


class PolicyTools(Protocol):
    async def search_policies(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        query_embedding: list[float],
    ) -> PolicyCandidates: ...


class ModerationState(TypedDict, total=False):
    request: ModerationAnalysisRequest
    input_validated: bool
    plan: PolicySearchPlan
    query_embedding: list[float]
    policies: PolicyCandidates
    decision: ModerationDecision
    decision_validated: bool
    response: ModerationAnalysisResponse


def _validate_grounding(state: ModerationState) -> None:
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
            or grounding.policyQuote not in policy.content
        ):
            raise ModelOutputError("신고 분석의 정책 근거를 확인할 수 없습니다.")
        policy_chunks.add(grounding.policyChunkId)

    if decision.recommendedAction == "FORCE_DELETE_MEETING" and request.targetType != "MEETING":
        raise ModelOutputError("신고 대상과 추천 제재가 일치하지 않습니다.")
    if decision.riskLevel == "CRITICAL" and decision.recommendedAction in {"DISMISS", "WARNING"}:
        raise ModelOutputError("위험도와 추천 제재가 일치하지 않습니다.")


def build_moderation_graph(model: ModerationModel, tools: PolicyTools):
    def validate_input(state: ModerationState):
        return {"input_validated": True}

    async def summarize_report(state: ModerationState):
        return {"plan": await model.prepare_moderation(state["request"])}

    async def embed_policy_query(state: ModerationState):
        return {"query_embedding": await model.embed(state["plan"].semanticQuery)}

    async def retrieve_policies(state: ModerationState):
        policies = await tools.search_policies(state["request"], state["plan"], state["query_embedding"])
        if not policies.items:
            raise ModelOutputError("신고 분석에 적용할 운영 정책을 찾지 못했습니다.")
        return {"policies": policies}

    async def classify_report(state: ModerationState):
        return {
            "decision": await model.analyze_moderation(
                state["request"], state["plan"], state["policies"].items
            )
        }

    def validate_evidence_and_policy(state: ModerationState):
        _validate_grounding(state)
        return {"decision_validated": True}

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
        ("classify_report", classify_report),
        ("validate_evidence_and_policy", validate_evidence_and_policy),
        ("build_result", build_result),
    ]
    for name, node in nodes:
        graph.add_node(name, node)
    graph.add_edge(START, "validate_input")
    for (current, _), (following, _) in zip(nodes, nodes[1:]):
        graph.add_edge(current, following)
    graph.add_edge("build_result", END)
    return graph.compile()
