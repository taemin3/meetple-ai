import json
from contextlib import asynccontextmanager

import httpx
import pytest
from conftest import FakeModel
from pydantic import SecretStr, ValidationError

from meetple_ai.app import create_app
from meetple_ai.backend import BackendClient, BackendRejected
from meetple_ai.contracts import (
    EvidenceGrounding,
    ModerationDecision,
    PolicyGrounding,
    PolicySearchPlan,
    Recommendation,
)
from meetple_ai.mcp_tools import connect_tools
from meetple_ai.settings import Settings

TOKEN = "test-service-000000000000000000000"
CAPABILITY = "test-signed-capability"


class FakeModerationModel:
    def __init__(self):
        self.embedded_texts = []

    async def prepare_moderation(self, request):
        return PolicySearchPlan(
            summary="반복적인 모욕 메시지 신고입니다.",
            keyword="반복 모욕",
            semanticQuery="채팅에서 상대방을 반복적으로 모욕하는 행위",
        )

    async def embed(self, semantic_query):
        return [0.01] * 1536

    async def embed_many(self, texts):
        self.embedded_texts.extend(texts)
        return [[0.01] * 1536 for _ in texts]

    async def analyze_moderation(self, request, plan, policies):
        return ModerationDecision(
            reportType="ABUSE_OR_HARASSMENT",
            riskLevel="MEDIUM",
            priority="HIGH",
            rationale="증거 메시지가 괴롭힘 방지 정책에 해당합니다.",
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


async def test_fastapi_graph_mcp_and_backend_contract(request_data, intent, candidate):
    """실제 MCP와 LangGraph를 사용하고 외부 LLM·Spring HTTP만 대체한다."""
    seen = []

    async def handle_backend(request):
        assert request.headers["X-Meetple-Capability"] == CAPABILITY
        assert request.headers["X-AI-Service-Token"] == TOKEN
        assert "authorization" not in request.headers
        seen.append(request.url.path)
        if request.url.path.endswith("categories"):
            data = ["운동"]
        else:
            body = json.loads(request.content)
            assert len(body["queryEmbedding"]) == 1536
            assert body["queryEmbeddingModel"] == "test-embedding-model"
            data = {"items": [candidate.model_dump(mode="json")], "hasMore": False}
        return httpx.Response(200, json={"success": True, "data": data})

    settings = Settings(service_token=SecretStr(TOKEN), _env_file=None)
    model = FakeModel(intent, [Recommendation(meetingId=10, evidenceQuote="처음 달리는 분 환영")])
    async with httpx.AsyncClient(
        base_url="http://backend", transport=httpx.MockTransport(handle_backend)
    ) as client:
        backend = BackendClient(client, TOKEN, "test-embedding-model")

        def factory(**kwargs):
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

        @asynccontextmanager
        async def local_tools(url, service_token, capability):
            async with connect_tools(
                "http://127.0.0.1:8001/mcp/", service_token, capability, factory
            ) as tools:
                yield tools

        app = create_app(settings, backend=backend, model=model, tools_factory=local_tools)
        async with app.router.lifespan_context(app):
            async with local_tools(None, TOKEN, CAPABILITY) as tools:
                tool_list = await tools.session.list_tools()
                assert {t.name for t in tool_list.tools} == {"list_categories", "search_meetings"}
                assert "capability" not in str([t.inputSchema for t in tool_list.tools])
                assert await tools.categories() == ["운동"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8001"
            ) as api:
                response = await api.post(
                    "/v1/search",
                    json=request_data.model_dump(mode="json"),
                    headers={
                        "X-AI-Service-Token": TOKEN,
                        "X-Meetple-Capability": CAPABILITY,
                    },
                )
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "COMPLETED"
                assert response.json()["recommendations"][0]["meetingId"] == 10
        assert seen == [
            "/internal/ai/search/categories",
            "/internal/ai/search/categories",
            "/internal/ai/search/meetings",
        ]


async def test_internal_api_and_mcp_reject_missing_service_auth(request_data):
    app = create_app(Settings(service_token=SecretStr(TOKEN), _env_file=None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        for path in (
            "/v1/search",
            "/v1/moderation/analyze",
            "/v1/moderation/policies/embeddings/sync",
            "/mcp/",
        ):
            response = await client.post(path, json=request_data.model_dump(mode="json"))
            assert response.status_code == 403
        assert (await client.get("/readyz")).status_code == 503


async def test_backend_rejects_unauthorized_envelope_without_leaking_body():
    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(lambda request: httpx.Response(403, json={"secret": "must-not-leak"})),
    ) as client:
        with pytest.raises(BackendRejected, match="백엔드가 AI 서비스 요청을 거부했습니다") as error:
            await BackendClient(client, TOKEN, "test-embedding-model").categories(CAPABILITY)
        assert "must-not-leak" not in str(error.value)


async def test_moderation_api_searches_spring_policy_without_user_capability():
    seen = []

    async def handle_backend(request):
        assert request.url.path == "/internal/ai/moderation/policies/search"
        assert request.headers["X-AI-Service-Token"] == TOKEN
        assert "x-meetple-capability" not in request.headers
        body = json.loads(request.content)
        assert body["targetType"] == "CHAT_MESSAGE"
        assert body["policyType"] is None
        assert body["queryEmbeddingModel"] == "text-embedding-3-small"
        assert body["limit"] == 5
        assert len(body["queryEmbedding"]) == 1536
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "items": [
                        {
                            "policyId": 11,
                            "policyChunkId": 101,
                            "policyCode": "COMMUNITY-ABUSE",
                            "policyTitle": "괴롭힘 방지 정책",
                            "policyType": "ABUSE_OR_HARASSMENT",
                            "targetType": "CHAT_MESSAGE",
                            "policyVersion": 1,
                            "clauseCode": "ABUSE-1",
                            "content": "반복적인 모욕이나 괴롭힘을 금지합니다.",
                            "contentHash": "a" * 64,
                            "effectiveFrom": "2026-01-01",
                            "effectiveTo": None,
                            "keywordMatched": True,
                            "semanticDistance": 0.1,
                            "hybridScore": 0.9,
                        }
                    ],
                    "hasMore": False,
                },
            },
        )

    settings = Settings(
        service_token=SecretStr(TOKEN),
        openai_embedding_model="text-embedding-3-small",
        _env_file=None,
    )
    async with httpx.AsyncClient(
        base_url="http://backend", transport=httpx.MockTransport(handle_backend)
    ) as client:
        backend = BackendClient(client, TOKEN, settings.openai_embedding_model)
        app = create_app(settings, backend=backend, model=FakeModerationModel())
        request_body = {
            "reportId": 77,
            "targetType": "CHAT_MESSAGE",
            "reason": "ABUSE_OR_HARASSMENT",
            "description": None,
            "evidence": [
                {
                    "evidenceId": 501,
                    "evidenceType": "CHAT_MESSAGE",
                    "content": "상대방을 반복해서 모욕하는 메시지",
                }
            ],
        }
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8001"
            ) as api:
                response = await api.post(
                    "/v1/moderation/analyze",
                    json=request_body,
                    headers={"X-AI-Service-Token": TOKEN},
                )

    assert response.status_code == 200, response.text
    assert response.json()["evidenceIds"] == [501]
    assert response.json()["policyIds"] == [11]
    assert response.json()["recommendedAction"] == "WARNING"
    assert len(seen) == 1


def test_embedding_model_is_fixed_to_text_embedding_3_small():
    with pytest.raises(ValidationError):
        Settings(openai_embedding_model="other-model", _env_file=None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"moderation_policy_min_hybrid_score": -0.01},
        {"moderation_policy_min_hybrid_score": 1.01},
        {"moderation_policy_result_limit": 0},
        {"moderation_policy_result_limit": 11},
    ],
)
def test_moderation_policy_retrieval_settings_are_bounded(overrides):
    with pytest.raises(ValidationError):
        Settings(**overrides, _env_file=None)


async def test_policy_embedding_sync_reads_jobs_and_upserts_fixed_model_vectors():
    requests = []

    async def handle_backend(request):
        requests.append((request.method, request.url.path))
        assert request.headers["X-AI-Service-Token"] == TOKEN
        assert "x-meetple-capability" not in request.headers
        if request.method == "GET":
            assert request.url.params["embeddingModel"] == "text-embedding-3-small"
            assert request.url.params["limit"] == "2"
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "items": [
                            {
                                "policyId": 11,
                                "policyChunkId": 101,
                                "policyCode": "COMMUNITY-ABUSE",
                                "policyVersion": 1,
                                "clauseCode": "ABUSE-1",
                                "content": "반복적인 모욕이나 괴롭힘을 금지합니다.",
                                "contentHash": "a" * 64,
                            }
                        ]
                    },
                },
            )
        body = json.loads(request.content)
        assert body["embeddingModel"] == "text-embedding-3-small"
        assert body["contentHash"] == "a" * 64
        assert len(body["embedding"]) == 1536
        return httpx.Response(200, json={"success": True})

    settings = Settings(service_token=SecretStr(TOKEN), _env_file=None)
    model = FakeModerationModel()
    async with httpx.AsyncClient(
        base_url="http://backend", transport=httpx.MockTransport(handle_backend)
    ) as client:
        app = create_app(
            settings,
            backend=BackendClient(client, TOKEN, settings.openai_embedding_model),
            model=model,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8001"
            ) as api:
                response = await api.post(
                    "/v1/moderation/policies/embeddings/sync",
                    json={"limit": 2},
                    headers={"X-AI-Service-Token": TOKEN},
                )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "requestedCount": 1,
        "embeddedCount": 1,
        "embeddingModel": "text-embedding-3-small",
    }
    assert model.embedded_texts == ["반복적인 모욕이나 괴롭힘을 금지합니다."]
    assert requests == [
        ("GET", "/internal/ai/moderation/policies/embedding-jobs"),
        ("PUT", "/internal/ai/moderation/policies/chunks/101/embedding"),
    ]
