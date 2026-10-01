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
from meetple_ai.contracts import (
    ModerationAnalysisRequest,
    ModerationAnalysisResponse,
    PolicyEmbeddingSyncRequest,
    PolicyEmbeddingSyncResponse,
    SearchRequest,
    SearchResponse,
)
from meetple_ai.graph import build_graph
from meetple_ai.mcp_tools import build_mcp, connect_tools
from meetple_ai.model import OpenAISearchModel
from meetple_ai.moderation_graph import build_moderation_graph
from meetple_ai.moderation_worker import ModerationAnalysisProcessor, ReportAnalysisKafkaWorker
from meetple_ai.settings import Settings

logger = logging.getLogger("meetple_ai")


def create_app(settings: Settings | None = None, *, backend=None, model=None, tools_factory=connect_tools):
    settings = settings or Settings()
    backend_http = httpx.AsyncClient(base_url=settings.backend_url, timeout=5, trust_env=False)
    backend = backend or BackendClient(
        backend_http,
        settings.service_token.get_secret_value(),
        settings.openai_embedding_model,
    )
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
    moderation_worker = None
    if settings.kafka_consumer_enabled and model is not None:
        moderation_worker = ReportAnalysisKafkaWorker(
            settings,
            ModerationAnalysisProcessor(backend, model, slots),
            backend,
        )

    @asynccontextmanager
    async def lifespan(app):
        worker_task = (
            asyncio.create_task(moderation_worker.run(), name="report-analysis-consumer")
            if moderation_worker
            else None
        )
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            if moderation_worker and worker_task:
                await moderation_worker.stop()
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            await backend_http.aclose()
            if openai_client:
                await openai_client.close()

    app = FastAPI(title="Meetple AI Service", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def internal_auth(request: Request, call_next):
        if request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)
        expected = settings.service_token.get_secret_value()
        supplied = request.headers.get("X-AI-Service-Token", "")
        capability = request.headers.get("X-Meetple-Capability", "")
        capability_required = request.url.path == "/v1/search" or request.url.path.startswith("/mcp")
        if (
            len(expected) < 32
            or not secrets.compare_digest(expected.encode(), supplied.encode())
            or (capability_required and not 1 <= len(capability) <= 300)
        ):
            return JSONResponse({"message": "내부 AI 서비스 권한이 필요합니다."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI 기본 검증 오류는 원본 입력을 포함하므로 반환하지 않는다.
        return JSONResponse({"message": "AI 요청 입력이 올바르지 않습니다."}, status_code=422)

    @app.get("/healthz")
    async def health():
        return {"status": "UP"}

    @app.get("/readyz")
    async def ready():
        worker_ready = not settings.kafka_consumer_enabled or (
            moderation_worker is not None and moderation_worker.running
        )
        ready_now = model is not None and worker_ready
        return JSONResponse({"ready": ready_now}, status_code=200 if ready_now else 503)

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

    @app.post("/v1/moderation/analyze", response_model=ModerationAnalysisResponse)
    async def analyze_moderation(body: ModerationAnalysisRequest):
        if model is None:
            return JSONResponse({"message": "AI 모델 설정이 필요합니다."}, status_code=503)
        request_id, started = uuid4().hex, monotonic()
        try:
            async with asyncio.timeout(45):
                async with slots:
                    result = await build_moderation_graph(model, backend).ainvoke(
                        {"request": body}, {"recursion_limit": 12}
                    )
            logger.info(
                "moderation_complete request_id=%s duration_ms=%d status=COMPLETED",
                request_id,
                round((monotonic() - started) * 1000),
            )
            return result["response"]
        except Exception as exc:
            # 신고 원문, 증거, 정책 원문, 인증값과 공급자 오류 본문은 로그에 포함하지 않는다.
            logger.warning(
                "moderation_failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return JSONResponse({"message": "AI 신고 분석을 완료하지 못했습니다."}, status_code=503)

    @app.post(
        "/v1/moderation/policies/embeddings/sync",
        response_model=PolicyEmbeddingSyncResponse,
    )
    async def sync_policy_embeddings(body: PolicyEmbeddingSyncRequest):
        if model is None:
            return JSONResponse({"message": "AI 모델 설정이 필요합니다."}, status_code=503)
        request_id, started = uuid4().hex, monotonic()
        try:
            async with asyncio.timeout(45):
                async with slots:
                    jobs = await backend.policy_embedding_jobs(body.limit)
                    if jobs.items:
                        embeddings = await model.embed_many([job.content for job in jobs.items])
                        for job, embedding in zip(jobs.items, embeddings, strict=True):
                            await backend.upsert_policy_embedding(job, embedding)
            logger.info(
                "policy_embedding_sync_complete request_id=%s duration_ms=%d count=%d",
                request_id,
                round((monotonic() - started) * 1000),
                len(jobs.items),
            )
            return PolicyEmbeddingSyncResponse(
                requestedCount=len(jobs.items),
                embeddedCount=len(jobs.items),
                embeddingModel=settings.openai_embedding_model,
            )
        except Exception as exc:
            # 정책 원문, 임베딩, 인증값과 공급자 오류 본문은 로그에 포함하지 않는다.
            logger.warning(
                "policy_embedding_sync_failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return JSONResponse(
                {"message": "운영 정책 임베딩 동기화를 완료하지 못했습니다."},
                status_code=503,
            )

    app.mount("/mcp", mcp.streamable_http_app())
    return app


app = create_app()
