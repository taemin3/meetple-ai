import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from meetple_ai.backend import BackendClient, BackendRejected, BackendUnavailable
from meetple_ai.contracts import ModerationAnalysisResponse
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


def worker(processor=None, backend=None):
    settings = Settings(
        kafka_consumer_enabled=True,
        kafka_retry_delays_seconds=(0, 0, 0, 0),
        kafka_max_poll_interval_ms=60_000,
        _env_file=None,
    )
    return ReportAnalysisKafkaWorker(
        settings,
        processor or FakeProcessor(),
        backend or FakeBackend(),
        now_ms=lambda: 0,
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
