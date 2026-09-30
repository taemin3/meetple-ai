import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from time import monotonic
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from meetple_ai.backend import BackendClient
from meetple_ai.contracts import SearchRequest, SearchResponse
from meetple_ai.graph import build_graph
from meetple_ai.mcp_tools import build_mcp, connect_tools
from meetple_ai.model import OpenAISearchModel
from meetple_ai.settings import Settings

logger = logging.getLogger("meetple_ai")


def create_app(settings: Settings | None = None, *, backend=None, model=None, tools_factory=connect_tools):
    settings = settings or Settings()
    backend_http = httpx.AsyncClient(base_url=settings.backend_url, timeout=5, trust_env=False)
    backend = backend or BackendClient(backend_http, settings.service_token.get_secret_value())
    mcp = build_mcp(backend)
    openai_client = None
    if model is None and settings.ready:
        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=12,
            max_retries=0,
        )
        model = OpenAISearchModel(
            openai_client,
            settings.openai_model,
            settings.openai_embedding_model,
        )
    slots = asyncio.Semaphore(4)

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield
        await backend_http.aclose()
        if openai_client:
            await openai_client.close()

    app = FastAPI(title="Meetple AI Search", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def internal_auth(request: Request, call_next):
        if request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)
        expected = settings.service_token.get_secret_value()
        supplied = request.headers.get("X-AI-Service-Token", "")
        capability = request.headers.get("X-Meetple-Capability", "")
        if (
            len(expected) < 32
            or not secrets.compare_digest(expected.encode(), supplied.encode())
            or not 1 <= len(capability) <= 300
        ):
            return JSONResponse({"message": "내부 검색 권한이 필요합니다."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI 기본 검증 오류는 원본 입력을 포함하므로 반환하지 않는다.
        return JSONResponse({"message": "검색 입력이 올바르지 않습니다."}, status_code=422)

    @app.get("/healthz")
    async def health():
        return {"status": "UP"}

    @app.get("/readyz")
    async def ready():
        return JSONResponse({"ready": model is not None}, status_code=200 if model else 503)

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        if model is None:
            return JSONResponse({"message": "AI 모델 설정이 필요합니다."}, status_code=503)
        request_id, started = uuid4().hex, monotonic()
        try:
            async with asyncio.timeout(35):
                async with slots:
                    async with tools_factory(
                        settings.mcp_url,
                        settings.service_token.get_secret_value(),
                        request.headers["X-Meetple-Capability"],
                    ) as tools:
                        result = await build_graph(model, tools).ainvoke(
                            {"request": body}, {"recursion_limit": 10}
                        )
            logger.info(
                "search_complete request_id=%s duration_ms=%d status=%s",
                request_id,
                round((monotonic() - started) * 1000),
                result["response"].status,
            )
            return result["response"]
        except Exception as exc:
            # 인증값, 질문, 문서, 공급자 오류 본문을 로그에 포함하지 않는다.
            logger.warning("search_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
            return JSONResponse({"message": "AI 검색을 완료하지 못했습니다."}, status_code=503)

    app.mount("/mcp", mcp.streamable_http_app())
    return app


app = create_app()
