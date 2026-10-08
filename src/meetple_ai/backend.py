import httpx
from pydantic import ValidationError

from meetple_ai.contracts import (
    Candidates,
    Filters,
    ModerationAnalysisRequest,
    ModerationAnalysisResponse,
    PolicyCandidates,
    PolicyEmbeddingJob,
    PolicyEmbeddingJobs,
    PolicySearchPlan,
)


class BackendUnavailable(Exception):
    pass


class BackendContractInvalid(Exception):
    pass


class BackendRejected(BackendUnavailable):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class BackendClient:
    def __init__(self, client: httpx.AsyncClient, service_token: str, embedding_model: str):
        self.client = client
        self.service_token = service_token
        self.embedding_model = embedding_model

    async def _request(self, method: str, path: str, capability: str | None = None, body=None, params=None):
        headers = {"X-AI-Service-Token": self.service_token}
        if capability is not None:
            headers["X-Meetple-Capability"] = capability
        try:
            response = await self.client.request(
                method,
                path,
                json=body,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            envelope = response.json()
            if not isinstance(envelope, dict) or envelope.get("success") is not True:
                raise ValueError("Invalid envelope")
            return envelope.get("data")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {408, 429}:
                raise BackendUnavailable("백엔드가 일시적으로 요청을 처리할 수 없습니다.") from exc
            if 400 <= exc.response.status_code < 500:
                raise BackendRejected(
                    "백엔드가 AI 서비스 요청을 거부했습니다.",
                    exc.response.status_code,
                ) from exc
            raise BackendUnavailable("모임 정보를 조회할 수 없습니다.") from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailable("모임 정보를 조회할 수 없습니다.") from exc
        except ValueError as exc:
            raise BackendContractInvalid("백엔드 응답 형식이 올바르지 않습니다.") from exc

    async def categories(self, capability: str) -> list[str]:
        data = await self._request("GET", "/internal/ai/search/categories", capability)
        if not isinstance(data, list) or not all(isinstance(value, str) for value in data):
            raise BackendContractInvalid("카테고리 응답이 올바르지 않습니다.")
        return data

    async def search(
        self, capability: str, filters: Filters, query_embedding: list[float] | None
    ) -> Candidates:
        body = filters.model_dump(mode="json")
        body["queryEmbedding"] = query_embedding
        body["queryEmbeddingModel"] = self.embedding_model if query_embedding is not None else None
        data = await self._request("POST", "/internal/ai/search/meetings", capability, body)
        return Candidates.model_validate(data)

    async def search_policies(
        self,
        request: ModerationAnalysisRequest,
        plan: PolicySearchPlan,
        query_embedding: list[float],
        limit: int,
    ) -> PolicyCandidates:
        body = {
            "keyword": plan.keyword,
            "targetType": request.targetType,
            # 초기 LLM 분류를 하드 필터로 신뢰하지 않고 모든 관련 유형을 검색한다.
            "policyType": None,
            "queryEmbedding": query_embedding,
            "queryEmbeddingModel": self.embedding_model,
            "limit": limit,
        }
        try:
            data = await self._request("POST", "/internal/ai/moderation/policies/search", body=body)
            return PolicyCandidates.model_validate(data)
        except BackendContractInvalid:
            raise
        except BackendRejected:
            raise
        except BackendUnavailable as exc:
            raise BackendUnavailable("운영 정책을 조회할 수 없습니다.") from exc
        except ValidationError as exc:
            raise BackendContractInvalid("운영 정책 응답이 올바르지 않습니다.") from exc

    async def policy_embedding_jobs(self, limit: int) -> PolicyEmbeddingJobs:
        try:
            data = await self._request(
                "GET",
                "/internal/ai/moderation/policies/embedding-jobs",
                params={"embeddingModel": self.embedding_model, "limit": limit},
            )
            return PolicyEmbeddingJobs.model_validate(data)
        except (BackendUnavailable, ValueError) as exc:
            raise BackendUnavailable("운영 정책 임베딩 작업을 조회할 수 없습니다.") from exc

    async def upsert_policy_embedding(self, job: PolicyEmbeddingJob, embedding: list[float]) -> None:
        body = {
            "embeddingModel": self.embedding_model,
            "contentHash": job.contentHash,
            "embedding": embedding,
        }
        try:
            await self._request(
                "PUT",
                f"/internal/ai/moderation/policies/chunks/{job.policyChunkId}/embedding",
                body=body,
            )
        except BackendUnavailable as exc:
            raise BackendUnavailable("운영 정책 임베딩을 저장할 수 없습니다.") from exc

    async def moderation_context(self, report_id: int) -> ModerationAnalysisRequest:
        try:
            data = await self._request(
                "GET",
                f"/internal/ai/moderation/reports/{report_id}/context",
            )
            context = ModerationAnalysisRequest.model_validate(data)
            if context.reportId != report_id:
                raise BackendContractInvalid("신고 분석 문맥의 식별자가 일치하지 않습니다.")
            return context
        except ValidationError as exc:
            raise BackendContractInvalid("신고 분석 문맥 응답이 올바르지 않습니다.") from exc

    async def complete_moderation(self, result: ModerationAnalysisResponse) -> None:
        body = result.model_dump(mode="json", exclude={"reportId"})
        try:
            await self._request(
                "PUT",
                f"/internal/ai/moderation/reports/{result.reportId}/analysis",
                body=body,
            )
        except BackendRejected as exc:
            if exc.status_code == 409:
                return
            raise

    async def fail_moderation(self, report_id: int, *, retryable: bool, failure_code: str) -> None:
        await self._request(
            "PUT",
            f"/internal/ai/moderation/reports/{report_id}/analysis/failure",
            body={"retryable": retryable, "failureCode": failure_code},
        )
