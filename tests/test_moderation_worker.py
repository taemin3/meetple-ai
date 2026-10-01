import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from meetple_ai.backend import (
    BackendClient,
    BackendContractInvalid,
    BackendRejected,
    BackendUnavailable,
)
from meetple_ai.contracts import (
    ModerationAnalysisRequest,
    ModerationAnalysisResponse,
    PolicySearchPlan,
)
from meetple_ai.model import ModelOutputError
from meetple_ai.moderation_worker import (
    FAILURE_CODE_HEADER,
    RETRY_AT_HEADER,
    ReportAnalysisKafkaWorker,
)
from meetple_ai.settings import Settings


class FakeProcessor:
    def __init__(self, error=None):
        self.error = error
        self.report_ids = []

    async def process(self, report_id):
        self.report_ids.append(report_id)
        if self.error:
            raise self.error


class FakeBackend:
    def __init__(self):
        self.failures = []

    async def fail_moderation(self, report_id, *, retryable, failure_code):
        self.failures.append((report_id, retryable, failure_code))


class FakeProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, **kwargs):
        self.sent.append((topic, kwargs))


class RejectedFailureBackend(FakeBackend):
    def __init__(self, status_code):
        super().__init__()
        self.status_code = status_code

    async def fail_moderation(self, report_id, *, retryable, failure_code):
        raise BackendRejected("rejected", self.status_code)


class UnavailableFailureBackend(FakeBackend):
    async def fail_moderation(self, report_id, *, retryable, failure_code):
        raise BackendUnavailable("backend down")


def worker(processor=None, backend=None, *, retry_delays=(0, 0, 0, 0), now_ms=lambda: 0):
    settings = Settings(
        kafka_consumer_enabled=True,
        kafka_retry_delays_seconds=retry_delays,
        kafka_max_poll_interval_ms=(max(retry_delays) + 60) * 1000,
        _env_file=None,
    )
    return ReportAnalysisKafkaWorker(
        settings,
        processor or FakeProcessor(),
        backend or FakeBackend(),
        now_ms=now_ms,
    )


def event(report_id=10, aggregate_id="10"):
    return {
        "eventId": str(uuid4()),
        "eventType": "REPORT_ANALYSIS_REQUESTED",
        "schemaVersion": 1,
        "occurredAt": "2026-10-01T10:00:00Z",
        "aggregateType": "report",
        "aggregateId": aggregate_id,
        "data": {"reportId": report_id},
    }


def message(topic="meetple.moderation.report-analysis.v1", payload=None, headers=None):
    return SimpleNamespace(
        topic=topic,
        partition=0,
        offset=1,
        key=b"report:10",
        value=json.dumps(payload or event()).encode(),
        headers=headers or [],
    )


@pytest.mark.asyncio
async def test_main_topic_processes_report_without_republishing():
    processor, backend, producer = FakeProcessor(), FakeBackend(), FakeProducer()

    await worker(processor, backend).handle(message(), producer)

    assert processor.report_ids == [10]
    assert backend.failures == []
    assert producer.sent == []


@pytest.mark.asyncio
async def test_transient_failure_moves_message_to_first_retry_topic():
    processor = FakeProcessor(BackendUnavailable("backend down"))
    backend, producer = FakeBackend(), FakeProducer()

    await worker(processor, backend).handle(message(), producer)

    assert backend.failures == [(10, True, "BACKEND_UNAVAILABLE")]
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.retry-0"
    headers = dict(producer.sent[0][1]["headers"])
    assert headers[RETRY_AT_HEADER] == b"0"
    assert headers[FAILURE_CODE_HEADER] == b"BACKEND_UNAVAILABLE"


@pytest.mark.asyncio
async def test_failure_callback_outage_does_not_block_retry_publish():
    processor = FakeProcessor(BackendUnavailable("backend down"))
    producer = FakeProducer()

    await worker(processor, UnavailableFailureBackend()).handle(message(), producer)

    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.retry-0"


@pytest.mark.asyncio
async def test_last_retry_failure_moves_message_to_dlq_and_becomes_permanent():
    processor = FakeProcessor(ModelOutputError("invalid output"))
    backend, producer = FakeBackend(), FakeProducer()
    retry_message = message(
        topic="meetple.moderation.report-analysis.v1.retry-3",
        headers=[(RETRY_AT_HEADER, b"0")],
    )

    await worker(processor, backend).handle(retry_message, producer)

    assert backend.failures == [(10, False, "MODEL_OUTPUT_INVALID")]
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.dlq"


@pytest.mark.asyncio
async def test_invalid_event_goes_directly_to_dlq_without_backend_callback():
    backend, producer = FakeBackend(), FakeProducer()
    invalid = message(payload=event(report_id=10, aggregate_id="11"))

    await worker(backend=backend).handle(invalid, producer)

    assert backend.failures == []
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.dlq"
    assert dict(producer.sent[0][1]["headers"])[FAILURE_CODE_HEADER] == b"INVALID_EVENT"


@pytest.mark.asyncio
async def test_retry_without_due_header_becomes_permanent_and_moves_to_dlq():
    backend, producer = FakeBackend(), FakeProducer()
    retry_message = message(topic="meetple.moderation.report-analysis.v1.retry-0")

    await worker(backend=backend).handle(retry_message, producer)

    assert backend.failures == [(10, False, "INVALID_RETRY_METADATA")]
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.dlq"
    assert dict(producer.sent[0][1]["headers"])[FAILURE_CODE_HEADER] == b"INVALID_RETRY_METADATA"


@pytest.mark.asyncio
async def test_retry_timestamp_beyond_configured_delay_moves_to_dlq():
    backend, producer = FakeBackend(), FakeProducer()
    retry_message = message(
        topic="meetple.moderation.report-analysis.v1.retry-0",
        headers=[(RETRY_AT_HEADER, b"10001")],
    )

    await worker(backend=backend, retry_delays=(5, 30, 120, 600)).handle(retry_message, producer)

    assert backend.failures == [(10, False, "INVALID_RETRY_METADATA")]
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.dlq"


@pytest.mark.asyncio
async def test_backend_contract_failure_is_permanent_without_retrying_analysis():
    processor = FakeProcessor(BackendContractInvalid("invalid policy response"))
    backend, producer = FakeBackend(), FakeProducer()

    await worker(processor, backend).handle(message(), producer)

    assert backend.failures == [(10, False, "BACKEND_CONTRACT_INVALID")]
    assert producer.sent[0][0] == "meetple.moderation.report-analysis.v1.dlq"


@pytest.mark.asyncio
async def test_completed_analysis_conflict_is_committed_without_duplicate_retry():
    processor = FakeProcessor(BackendUnavailable("late duplicate"))
    producer = FakeProducer()

    await worker(processor, RejectedFailureBackend(409)).handle(message(), producer)

    assert producer.sent == []


@pytest.mark.asyncio
async def test_auth_rejection_does_not_commit_or_republish_message():
    processor = FakeProcessor(BackendUnavailable("backend down"))
    producer = FakeProducer()

    with pytest.raises(BackendRejected):
        await worker(processor, RejectedFailureBackend(403)).handle(message(), producer)

    assert producer.sent == []


@pytest.mark.asyncio
async def test_backend_context_and_callbacks_match_spring_contract():
    seen = []

    async def handle(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "GET":
            data = {
                "reportId": 10,
                "targetType": "CHAT_MESSAGE",
                "reason": "SPAM",
                "description": None,
                "evidence": [
                    {
                        "evidenceId": 10,
                        "evidenceType": "CHAT_MESSAGE",
                        "content": "반복 광고",
                    }
                ],
            }
        else:
            data = None
        return httpx.Response(200, json={"success": True, "data": data})

    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(handle),
    ) as client:
        backend = BackendClient(client, "service-token", "text-embedding-3-small")
        context = await backend.moderation_context(10)
        result = ModerationAnalysisResponse(
            reportId=10,
            reportType="SPAM",
            riskLevel="LOW",
            priority="LOW",
            summary="광고 신고",
            rationale="증거와 정책에 따른 판단",
            evidenceIds=[10],
            policyIds=[40],
            confidence=0.987654,
            recommendedAction="WARNING",
        )
        await backend.complete_moderation(result)
        await backend.fail_moderation(10, retryable=True, failure_code="ANALYSIS_TIMEOUT")

    assert context.reportId == 10
    assert seen[1] == (
        "PUT",
        "/internal/ai/moderation/reports/10/analysis",
        {
            "reportType": "SPAM",
            "riskLevel": "LOW",
            "priority": "LOW",
            "summary": "광고 신고",
            "rationale": "증거와 정책에 따른 판단",
            "evidenceIds": [10],
            "policyIds": [40],
            "confidence": 0.9877,
            "recommendedAction": "WARNING",
        },
    )
    assert seen[2] == (
        "PUT",
        "/internal/ai/moderation/reports/10/analysis/failure",
        {"retryable": True, "failureCode": "ANALYSIS_TIMEOUT"},
    )


@pytest.mark.asyncio
async def test_backend_rejects_mismatched_context_report_id():
    async def handle(request):
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "reportId": 11,
                    "targetType": "CHAT_MESSAGE",
                    "reason": "SPAM",
                    "description": None,
                    "evidence": [
                        {
                            "evidenceId": 10,
                            "evidenceType": "CHAT_MESSAGE",
                            "content": "반복 광고",
                        }
                    ],
                },
            },
        )

    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(handle),
    ) as client:
        backend = BackendClient(client, "service-token", "text-embedding-3-small")
        with pytest.raises(BackendContractInvalid, match="식별자가 일치하지 않습니다"):
            await backend.moderation_context(10)


@pytest.mark.asyncio
async def test_backend_preserves_invalid_policy_response_as_contract_failure():
    async def handle(request):
        return httpx.Response(
            200,
            json={"success": True, "data": {"items": [{"policyId": 1}], "hasMore": False}},
        )

    request = ModerationAnalysisRequest(
        reportId=10,
        targetType="CHAT_MESSAGE",
        reason="SPAM",
        evidence=[{"evidenceId": 10, "evidenceType": "CHAT_MESSAGE", "content": "반복 광고"}],
    )
    plan = PolicySearchPlan(summary="광고 신고", keyword="광고", semanticQuery="반복 광고")
    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(handle),
    ) as client:
        backend = BackendClient(client, "service-token", "text-embedding-3-small")
        with pytest.raises(BackendContractInvalid, match="운영 정책 응답이 올바르지 않습니다"):
            await backend.search_policies(request, plan, [0.0] * 1536)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429])
async def test_backend_treats_timeout_and_rate_limit_as_transient(status_code):
    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code)),
    ) as client:
        backend = BackendClient(client, "service-token", "text-embedding-3-small")
        with pytest.raises(BackendUnavailable) as error:
            await backend.moderation_context(10)
        assert not isinstance(error.value, BackendRejected)
