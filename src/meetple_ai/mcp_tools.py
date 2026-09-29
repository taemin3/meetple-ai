from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from meetple_ai.backend import BackendClient, BackendUnavailable
from meetple_ai.contracts import Candidates, Filters


def build_mcp(backend: BackendClient) -> FastMCP:
    server = FastMCP(
        "Meetple meeting search",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        max_request_body_size=16384,
    )
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    def capability(ctx: Context) -> str:
        # 인증값을 도구 입력 스키마/LLM 프롬프트에 노출하지 않는다.
        request = ctx.request_context.request
        value = request.headers.get("X-Meetple-Capability") if request else None
        if not value:
            raise BackendUnavailable("검색 권한이 필요합니다.")
        return value

    @server.tool(annotations=annotations)
    async def list_categories(ctx: Context) -> dict[str, list[str]]:
        """Meetple에서 사용 가능한 모임 카테고리를 조회합니다."""
        return {"categories": await backend.categories(capability(ctx))}

    @server.tool(annotations=annotations)
    async def search_meetings(filters: Filters, ctx: Context) -> Candidates:
        """날짜·거리·모집 상태·차단 규칙을 적용해 모임 후보를 조회합니다. 날짜 끝은 미포함입니다."""
        return await backend.search(capability(ctx), filters)

    return server


class McpMeetingTools:
    def __init__(self, session: ClientSession):
        self.session = session

    async def _call(self, name: str, arguments: dict) -> dict:
        result = await self.session.call_tool(name, arguments)
        if result.isError or not isinstance(result.structuredContent, dict):
            raise BackendUnavailable("검색 도구 실행에 실패했습니다.")
        return result.structuredContent

    async def categories(self) -> list[str]:
        return (await self._call("list_categories", {}))["categories"]

    async def search(self, filters: Filters) -> Candidates:
        result = await self._call("search_meetings", {"filters": filters.model_dump(mode="json")})
        return Candidates.model_validate(result)


@asynccontextmanager
async def connect_tools(url: str, service_token: str, capability: str, client_factory=None):
    factory = client_factory or httpx.AsyncClient
    async with factory(
        headers={"X-AI-Service-Token": service_token, "X-Meetple-Capability": capability},
        timeout=httpx.Timeout(8, read=10),
        trust_env=False,
    ) as client:
        async with streamable_http_client(url, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield McpMeetingTools(session)
