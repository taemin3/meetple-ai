import pytest

from meetple_ai.contracts import (
    EvidenceGrounding,
    ModerationAnalysisRequest,
    ModerationDecision,
    ModerationEvidence,
    PolicyCandidate,
    PolicyCandidates,
    PolicyGrounding,
    PolicySearchPlan,
)
from meetple_ai.model import ModelOutputError
from meetple_ai.moderation_graph import build_moderation_graph


def request_fixture():
    return ModerationAnalysisRequest(
        reportId=77,
        targetType="CHAT_MESSAGE",
        reason="ABUSE_OR_HARASSMENT",
        description=None,
        evidence=[
            ModerationEvidence(
                evidenceId=501,
                evidenceType="CHAT_MESSAGE",
                content="상대방을 반복해서 모욕하는 메시지",
            )
        ],
    )


def policy_fixture():
    return PolicyCandidate(
        policyId=11,
        policyChunkId=101,
        policyCode="COMMUNITY-ABUSE",
        policyTitle="괴롭힘 방지 정책",
        policyType="ABUSE_OR_HARASSMENT",
        targetType="CHAT_MESSAGE",
        policyVersion=1,
        clauseCode="ABUSE-1",
        content="반복적인 모욕이나 괴롭힘을 금지합니다.",
        contentHash="a" * 64,
        effectiveFrom="2026-01-01",
        effectiveTo=None,
        keywordMatched=True,
        semanticDistance=0.1,
        hybridScore=0.9,
    )


def decision_fixture(**updates):
    decision = ModerationDecision(
        reportType="ABUSE_OR_HARASSMENT",
        riskLevel="MEDIUM",
        priority="HIGH",
        rationale="반복적인 모욕 표현이 괴롭힘 방지 정책에 해당합니다.",
        evidence=[EvidenceGrounding(evidenceId=501, evidenceQuote="반복해서 모욕")],
        policies=[
            PolicyGrounding(
                policyId=11,
                policyChunkId=101,
                policyQuote="반복적인 모욕이나 괴롭힘",
            )
        ],
        confidence=0.88,
        recommendedAction="WARNING",
    )
    return decision.model_copy(update=updates)


class FakeModerationModel:
    def __init__(self, decision=None):
        self.plan = PolicySearchPlan(
            summary="채팅에서 상대방을 반복적으로 모욕했다는 신고입니다.",
            keyword="반복 모욕",
            semanticQuery="채팅에서 상대방을 반복적으로 모욕하고 괴롭히는 행위",
            policyType="ABUSE_OR_HARASSMENT",
        )
        self.decision = decision or decision_fixture()
        self.calls = []

    async def prepare_moderation(self, request):
        self.calls.append("prepare")
        return self.plan

    async def embed(self, semantic_query):
        self.calls.append(("embed", semantic_query))
        return [0.01] * 1536

    async def analyze_moderation(self, request, plan, policies):
        self.calls.append(("analyze", [policy.policyChunkId for policy in policies]))
        return self.decision


class FakePolicyTools:
    def __init__(self, policies=()):
        self.policies = list(policies)
        self.calls = []

    async def search_policies(self, request, plan, query_embedding):
        self.calls.append((request.reportId, plan.keyword, len(query_embedding)))
        return PolicyCandidates(items=self.policies, hasMore=False)


async def test_moderation_graph_returns_only_verified_ids():
    request = request_fixture()
    model = FakeModerationModel()
    tools = FakePolicyTools([policy_fixture()])

    result = await build_moderation_graph(model, tools).ainvoke({"request": request})

    assert result["response"].model_dump() == {
        "reportId": 77,
        "reportType": "ABUSE_OR_HARASSMENT",
        "riskLevel": "MEDIUM",
        "priority": "HIGH",
        "summary": "채팅에서 상대방을 반복적으로 모욕했다는 신고입니다.",
        "rationale": "반복적인 모욕 표현이 괴롭힘 방지 정책에 해당합니다.",
        "evidenceIds": [501],
        "policyIds": [11],
        "confidence": 0.88,
        "recommendedAction": "WARNING",
    }
    assert model.calls == [
        "prepare",
        ("embed", "채팅에서 상대방을 반복적으로 모욕하고 괴롭히는 행위"),
        ("analyze", [101]),
    ]
    assert tools.calls == [(77, "반복 모욕", 1536)]


@pytest.mark.parametrize(
    "decision",
    [
        decision_fixture(evidence=[EvidenceGrounding(evidenceId=999, evidenceQuote="반복해서 모욕")]),
        decision_fixture(evidence=[EvidenceGrounding(evidenceId=501, evidenceQuote="없는 증거")]),
        decision_fixture(
            policies=[
                PolicyGrounding(
                    policyId=999,
                    policyChunkId=101,
                    policyQuote="반복적인 모욕이나 괴롭힘",
                )
            ]
        ),
        decision_fixture(
            policies=[PolicyGrounding(policyId=11, policyChunkId=101, policyQuote="없는 정책 문구")]
        ),
    ],
)
async def test_moderation_graph_rejects_unverified_evidence_and_policy(decision):
    with pytest.raises(ModelOutputError):
        await build_moderation_graph(
            FakeModerationModel(decision), FakePolicyTools([policy_fixture()])
        ).ainvoke({"request": request_fixture()})


async def test_moderation_graph_fails_closed_without_policy_candidates():
    model = FakeModerationModel()
    with pytest.raises(ModelOutputError, match="운영 정책"):
        await build_moderation_graph(model, FakePolicyTools()).ainvoke({"request": request_fixture()})
    assert all(call != ("analyze", []) for call in model.calls)


async def test_force_delete_is_rejected_for_non_meeting_target():
    decision = decision_fixture(recommendedAction="FORCE_DELETE_MEETING")
    with pytest.raises(ModelOutputError, match="추천 제재"):
        await build_moderation_graph(
            FakeModerationModel(decision), FakePolicyTools([policy_fixture()])
        ).ainvoke({"request": request_fixture()})


def test_moderation_request_rejects_duplicate_or_excessive_evidence():
    request = request_fixture().model_dump()
    request["evidence"].append(request["evidence"][0])
    with pytest.raises(ValueError, match="중복"):
        ModerationAnalysisRequest.model_validate(request)

    request = request_fixture().model_dump()
    request["evidence"][0]["content"] = "가" * 4000
    request["evidence"].extend(
        [
            {"evidenceId": evidence_id, "evidenceType": "CHAT_MESSAGE", "content": "나" * 4000}
            for evidence_id in (502, 503, 504)
        ]
    )
    with pytest.raises(ValueError, match="전체 길이"):
        ModerationAnalysisRequest.model_validate(request)
