from contextlib import asynccontextmanager

import httpx
from conftest import FakeModel
from pydantic import SecretStr

from meetple_ai.app import create_app
from meetple_ai.backend import BackendClient
from meetple_ai.contracts import Recommendation
from meetple_ai.mcp_tools import connect_tools
from meetple_ai.settings import Settings

TOKEN = "test-service-000000000000000000000"
CAPABILITY = "test-signed-capability"


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
            data = {"items": [candidate.model_dump(mode="json")], "hasMore": False}
        return httpx.Response(200, json={"success": True, "data": data})

    settings = Settings(service_token=SecretStr(TOKEN), _env_file=None)
    model = FakeModel(intent, [Recommendation(meetingId=10, evidenceQuote="처음 달리는 분 환영")])
    async with httpx.AsyncClient(
        base_url="http://backend", transport=httpx.MockTransport(handle_backend)
    ) as client:
        backend = BackendClient(client, TOKEN)

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
        for path in ("/v1/search", "/mcp/"):
            response = await client.post(path, json=request_data.model_dump(mode="json"))
            assert response.status_code == 403
        assert (await client.get("/readyz")).status_code == 503


async def test_backend_rejects_unauthorized_envelope_without_leaking_body():
    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(lambda request: httpx.Response(403, json={"secret": "must-not-leak"})),
    ) as client:
        import pytest

        from meetple_ai.backend import BackendUnavailable

        with pytest.raises(BackendUnavailable, match="모임 정보를 조회할 수 없습니다") as error:
            await BackendClient(client, TOKEN).categories(CAPABILITY)
        assert "must-not-leak" not in str(error.value)
