import httpx

from meetple_ai.contracts import Candidates, Filters


class BackendUnavailable(Exception):
    pass


class BackendClient:
    def __init__(self, client: httpx.AsyncClient, service_token: str):
        self.client = client
        self.service_token = service_token

    async def _request(self, method: str, path: str, capability: str, body=None):
        try:
            response = await self.client.request(
                method,
                path,
                json=body,
                headers={"X-AI-Service-Token": self.service_token, "X-Meetple-Capability": capability},
            )
            response.raise_for_status()
            envelope = response.json()
            if envelope.get("success") is not True:
                raise ValueError("Invalid envelope")
            return envelope["data"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise BackendUnavailable("모임 정보를 조회할 수 없습니다.") from exc

    async def categories(self, capability: str) -> list[str]:
        data = await self._request("GET", "/internal/ai/search/categories", capability)
        if not isinstance(data, list) or not all(isinstance(value, str) for value in data):
            raise BackendUnavailable("카테고리 응답이 올바르지 않습니다.")
        return data

    async def search(
        self, capability: str, filters: Filters, query_embedding: list[float] | None
    ) -> Candidates:
        body = filters.model_dump(mode="json")
        body["queryEmbedding"] = query_embedding
        data = await self._request("POST", "/internal/ai/search/meetings", capability, body)
        return Candidates.model_validate(data)
