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
from meetple_ai.moderation_graph import build_moderation_graph, normalize_recommended_action


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
    def __init__(self, decision=None, repaired_decision=None):
        self.plan = PolicySearchPlan(
            summary="채팅에서 상대방을 반복적으로 모욕했다는 신고입니다.",
            keyword="반복 모욕",
            semanticQuery="채팅에서 상대방을 반복적으로 모욕하고 괴롭히는 행위",
        )
        self.refined_plan = self.plan.model_copy(
            update={
                "keyword": "언어 괴롭힘",
                "semanticQuery": "채팅에서 타인을 언어로 괴롭히는 행위",
            }
        )
        self.decision = decision or decision_fixture()
        self.repaired_decision = repaired_decision or self.decision
        self.calls = []

    async def prepare_moderation(self, request):
        self.calls.append("prepare")
        return self.plan

    async def embed(self, semantic_query):
        self.calls.append(("embed", semantic_query))
        return [0.01] * 1536

    async def refine_moderation_search(self, request, previous_plan):
        self.calls.append(("refine", previous_plan.keyword))
        return self.refined_plan

    async def analyze_moderation(self, request, plan, policies):
        self.calls.append(("analyze", [policy.policyChunkId for policy in policies]))
        return self.decision

    async def repair_moderation_decision(self, request, plan, policies, previous_decision, validation_error):
        self.calls.append(("repair", validation_error))
        return self.repaired_decision


class FakePolicyTools:
    def __init__(self, policies=(), *, responses=None):
        self.responses = [list(policies)] if responses is None else [list(items) for items in responses]
        self.calls = []

    async def search_policies(self, request, plan, query_embedding, limit):
        self.calls.append((request.reportId, plan.keyword, len(query_embedding), limit))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return PolicyCandidates(items=self.responses[index], hasMore=False)


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
    assert tools.calls == [(77, "반복 모욕", 1536, 5)]


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
    tools = FakePolicyTools()
    with pytest.raises(ModelOutputError, match="운영 정책"):
        await build_moderation_graph(model, tools).ainvoke({"request": request_fixture()})
    assert [call for call in model.calls if isinstance(call, tuple) and call[0] == "refine"] == [
        ("refine", "반복 모욕")
    ]
    assert len(tools.calls) == 2
    assert all(call != ("analyze", []) for call in model.calls)


async def test_moderation_graph_rewrites_search_once_when_no_candidate_meets_threshold():
    low_score = policy_fixture().model_copy(update={"keywordMatched": True, "hybridScore": 0.29})
    model = FakeModerationModel()
    tools = FakePolicyTools(responses=[[low_score], [policy_fixture()]])

    result = await build_moderation_graph(model, tools).ainvoke({"request": request_fixture()})

    assert result["response"].policyIds == [11]
    assert model.calls[:5] == [
        "prepare",
        ("embed", "채팅에서 상대방을 반복적으로 모욕하고 괴롭히는 행위"),
        ("refine", "반복 모욕"),
        ("embed", "채팅에서 타인을 언어로 괴롭히는 행위"),
        ("analyze", [101]),
    ]
    assert [call[1] for call in tools.calls] == ["반복 모욕", "언어 괴롭힘"]


async def test_moderation_graph_filters_minimum_relevance_and_limits_model_context():
    low_score = policy_fixture().model_copy(
        update={"policyId": 20, "policyChunkId": 200, "keywordMatched": True, "hybridScore": 0.29}
    )
    relevant = [
        policy_fixture(),
        *[
            policy_fixture().model_copy(
                update={
                    "policyId": policy_id,
                    "policyChunkId": policy_id * 10,
                    "keywordMatched": False,
                    "hybridScore": 0.30,
                }
            )
            for policy_id in range(21, 27)
        ],
    ]
    model = FakeModerationModel()

    await build_moderation_graph(model, FakePolicyTools([low_score, *relevant])).ainvoke(
        {"request": request_fixture()}
    )

    analyze_call = next(call for call in model.calls if isinstance(call, tuple) and call[0] == "analyze")
    assert analyze_call[1] == [101, 210, 220, 230, 240]


async def test_moderation_graph_repairs_invalid_grounding_once():
    invalid = decision_fixture(evidence=[EvidenceGrounding(evidenceId=501, evidenceQuote="없는 증거")])
    model = FakeModerationModel(invalid, repaired_decision=decision_fixture())

    result = await build_moderation_graph(model, FakePolicyTools([policy_fixture()])).ainvoke(
        {"request": request_fixture()}
    )

    assert result["response"].evidenceIds == [501]
    assert [call[0] for call in model.calls if isinstance(call, tuple)].count("repair") == 1


async def test_grounding_repair_cannot_raise_risk_priority_confidence_or_sanction():
    invalid = decision_fixture(
        evidence=[EvidenceGrounding(evidenceId=501, evidenceQuote="없는 증거")],
        confidence=0.70,
    )
    escalated_repair = decision_fixture(
        reportType="SAFETY",
        riskLevel="CRITICAL",
        priority="URGENT",
        rationale="보정 모델이 판단 수위를 높였습니다.",
        confidence=0.99,
        recommendedAction="PERMANENT_SUSPENSION",
    )
    model = FakeModerationModel(invalid, repaired_decision=escalated_repair)

    result = await build_moderation_graph(model, FakePolicyTools([policy_fixture()])).ainvoke(
        {"request": request_fixture()}
    )

    response = result["response"]
    assert response.reportType == "ABUSE_OR_HARASSMENT"
    assert response.riskLevel == "MEDIUM"
    assert response.priority == "HIGH"
    assert response.rationale == invalid.rationale
    assert response.confidence == 0.70
    assert response.recommendedAction == "WARNING"
    assert response.evidenceIds == [501]
    assert response.policyIds == [11]


async def test_force_delete_is_limited_to_manual_review_for_non_meeting_target():
    decision = decision_fixture(riskLevel="HIGH", recommendedAction="FORCE_DELETE_MEETING")
    result = await build_moderation_graph(
        FakeModerationModel(decision), FakePolicyTools([policy_fixture()])
    ).ainvoke({"request": request_fixture()})
    assert result["response"].recommendedAction == "MANUAL_REVIEW"


def test_all_risk_and_action_combinations_are_deterministically_limited():
    allowed = {
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
    actions = {
        "DISMISS",
        "WARNING",
        "SUSPEND_1_DAY",
        "SUSPEND_3_DAYS",
        "SUSPEND_7_DAYS",
        "PERMANENT_SUSPENSION",
        "FORCE_DELETE_MEETING",
        "MANUAL_REVIEW",
    }
    for risk_level, allowed_actions in allowed.items():
        for action in actions:
            decision = decision_fixture(riskLevel=risk_level, recommendedAction=action)
            actual = normalize_recommended_action(decision, "MEETING").recommendedAction
            assert actual == (action if action in allowed_actions else "MANUAL_REVIEW")


async def test_policy_quote_must_exist_in_exact_context_sent_to_model():
    hidden_quote = "모델에 전달되지 않은 정책 근거"
    policy = policy_fixture().model_copy(update={"content": "가" * 3000 + hidden_quote})
    decision = decision_fixture(
        policies=[PolicyGrounding(policyId=11, policyChunkId=101, policyQuote=hidden_quote)]
    )
    with pytest.raises(ModelOutputError, match="정책 근거"):
        await build_moderation_graph(FakeModerationModel(decision), FakePolicyTools([policy])).ainvoke(
            {"request": request_fixture()}
        )


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
