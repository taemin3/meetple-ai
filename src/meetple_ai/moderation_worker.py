import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition
from pydantic import ValidationError

from meetple_ai.backend import BackendContractInvalid, BackendRejected, BackendUnavailable
from meetple_ai.contracts import ReportAnalysisRequestedEnvelope
from meetple_ai.model import ModelOutputError
from meetple_ai.moderation_graph import build_moderation_graph
from meetple_ai.settings import Settings

logger = logging.getLogger("meetple_ai.moderation_worker")

RETRY_AT_HEADER = "meetpleRetryAtEpochMs"
FAILURE_CODE_HEADER = "meetpleFailureCode"
ORIGINAL_TOPIC_HEADER = "meetpleOriginalTopic"
RETRY_CLOCK_SKEW_SECONDS = 5


class InvalidModerationEvent(Exception):
    pass


class ModerationAnalysisProcessor:
    def __init__(
        self,
        backend,
        model,
        slots: asyncio.Semaphore,
        *,
        min_policy_hybrid_score: float = 0.30,
        policy_result_limit: int = 5,
    ):
        self.backend = backend
        self.model = model
        self.slots = slots
        self.min_policy_hybrid_score = min_policy_hybrid_score
        self.policy_result_limit = policy_result_limit

    async def process(self, report_id: int) -> None:
        request = await self.backend.moderation_context(report_id)
        async with self.slots:
            result = await build_moderation_graph(
                self.model,
                self.backend,
                min_policy_hybrid_score=self.min_policy_hybrid_score,
                policy_result_limit=self.policy_result_limit,
            ).ainvoke(
                {"request": request},
                {"recursion_limit": 20},
            )
        await self.backend.complete_moderation(result["response"])


class ReportAnalysisKafkaWorker:
    def __init__(
        self,
        settings: Settings,
        processor: ModerationAnalysisProcessor,
        backend,
        *,
        consumer_factory=AIOKafkaConsumer,
        producer_factory=AIOKafkaProducer,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_ms: Callable[[], int] = lambda: round(time.time() * 1000),
    ):
        self.settings = settings
        self.processor = processor
        self.backend = backend
        self.consumer_factory = consumer_factory
        self.producer_factory = producer_factory
        self.sleep = sleep
        self.now_ms = now_ms
        self._running = False
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def topics(self) -> tuple[str, ...]:
        base = self.settings.kafka_topic
        retries = tuple(
            f"{base}.retry-{index}" for index in range(len(self.settings.kafka_retry_delays_seconds))
        )
        return (base, *retries)

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._run_session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._running = False
                logger.warning("moderation_consumer_restart error_type=%s", type(exc).__name__)
                await self._wait_or_stop(5)

    async def stop(self) -> None:
        self._stop.set()

    async def _run_session(self) -> None:
        bootstrap_servers = [
            value.strip() for value in self.settings.kafka_bootstrap_servers.split(",") if value.strip()
        ]
        consumer = self.consumer_factory(
            *self.topics,
            bootstrap_servers=bootstrap_servers,
            group_id=self.settings.kafka_consumer_group,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_interval_ms=self.settings.kafka_max_poll_interval_ms,
        )
        producer = self.producer_factory(
            bootstrap_servers=bootstrap_servers,
            enable_idempotence=True,
        )
        await producer.start()
        resume_tasks: set[asyncio.Task] = set()
        try:
            await consumer.start()
            self._running = True
            logger.info("moderation_consumer_started topic=%s", self.settings.kafka_topic)
            async for message in consumer:
                if self._stop.is_set():
                    break
                partition = TopicPartition(message.topic, message.partition)
                try:
                    try:
                        retry_wait = self._retry_wait_seconds(message.topic, message.headers)
                    except InvalidModerationEvent:
                        retry_wait = 0
                    if retry_wait > 0:
                        consumer.seek(partition, message.offset)
                        consumer.pause(partition)
                        task = asyncio.create_task(self._resume_partition(consumer, partition, retry_wait))
                        resume_tasks.add(task)
                        task.add_done_callback(resume_tasks.discard)
                        continue
                    await self.handle(message, producer)
                    await consumer.commit({partition: message.offset + 1})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    consumer.seek(partition, message.offset)
                    logger.warning(
                        "moderation_message_deferred topic=%s partition=%d offset=%d error_type=%s",
                        message.topic,
                        message.partition,
                        message.offset,
                        type(exc).__name__,
                    )
                    await self._wait_or_stop(1)
        finally:
            self._running = False
            for task in resume_tasks:
                task.cancel()
            if resume_tasks:
                await asyncio.gather(*resume_tasks, return_exceptions=True)
            await consumer.stop()
            await producer.stop()

    async def handle(self, message: Any, producer: Any) -> None:
        attempt = self._attempt_for_topic(message.topic)
        try:
            envelope = self._parse_event(message.value)
        except InvalidModerationEvent:
            await self._publish_dlq(message, producer, "INVALID_EVENT")
            return

        if attempt >= 0:
            try:
                await self._wait_for_retry(message.topic, message.headers)
            except InvalidModerationEvent:
                notified = await self._notify_failure(
                    envelope.data.reportId,
                    retryable=False,
                    failure_code="INVALID_RETRY_METADATA",
                )
                if notified:
                    await self._publish_dlq(message, producer, "INVALID_RETRY_METADATA")
                return

        try:
            async with asyncio.timeout(60):
                await self.processor.process(envelope.data.reportId)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            retryable, failure_code = self._classify_failure(exc)
            if retryable and attempt < len(self.settings.kafka_retry_delays_seconds) - 1:
                notified = await self._notify_failure(
                    envelope.data.reportId,
                    retryable=True,
                    failure_code=failure_code,
                )
                if notified:
                    await self._publish_retry(message, producer, attempt + 1, failure_code)
                return
            notified = await self._notify_failure(
                envelope.data.reportId,
                retryable=False,
                failure_code=failure_code,
            )
            if notified:
                await self._publish_dlq(message, producer, failure_code)

    def _parse_event(self, value: bytes | bytearray | memoryview | str) -> ReportAnalysisRequestedEnvelope:
        try:
            payload = json.loads(value)
            return ReportAnalysisRequestedEnvelope.model_validate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValidationError) as exc:
            raise InvalidModerationEvent("신고 분석 이벤트가 올바르지 않습니다.") from exc

    def _attempt_for_topic(self, topic: str) -> int:
        if topic == self.settings.kafka_topic:
            return -1
        prefix = f"{self.settings.kafka_topic}.retry-"
        if topic.startswith(prefix):
            try:
                attempt = int(topic.removeprefix(prefix))
            except ValueError as exc:
                raise InvalidModerationEvent("Retry 토픽이 올바르지 않습니다.") from exc
            if 0 <= attempt < len(self.settings.kafka_retry_delays_seconds):
                return attempt
        raise InvalidModerationEvent("신고 분석 토픽이 올바르지 않습니다.")

    async def _wait_for_retry(
        self,
        topic: str,
        headers: list[tuple[str, bytes]] | None,
    ) -> None:
        wait_seconds = self._retry_wait_seconds(topic, headers)
        if wait_seconds:
            await self.sleep(wait_seconds)

    def _retry_wait_seconds(
        self,
        topic: str,
        headers: list[tuple[str, bytes]] | None,
    ) -> float:
        attempt = self._attempt_for_topic(topic)
        if attempt < 0:
            return 0
        retry_at_value = self._last_header(headers, RETRY_AT_HEADER)
        try:
            retry_at = int(retry_at_value.decode("ascii"))
        except (AttributeError, UnicodeDecodeError, ValueError) as exc:
            raise InvalidModerationEvent("Retry 시각이 올바르지 않습니다.") from exc
        remaining_ms = retry_at - self.now_ms()
        max_remaining_ms = (
            self.settings.kafka_retry_delays_seconds[attempt] + RETRY_CLOCK_SKEW_SECONDS
        ) * 1000
        if retry_at < 0 or remaining_ms > max_remaining_ms:
            raise InvalidModerationEvent("Retry 시각이 허용 범위를 벗어났습니다.")
        return max(0, remaining_ms) / 1000

    async def _resume_partition(self, consumer: Any, partition: TopicPartition, delay: float) -> None:
        await self.sleep(delay)
        if not self._stop.is_set():
            consumer.resume(partition)

    async def _notify_failure(
        self,
        report_id: int,
        *,
        retryable: bool,
        failure_code: str,
    ) -> bool:
        try:
            await self.backend.fail_moderation(
                report_id,
                retryable=retryable,
                failure_code=failure_code,
            )
            return True
        except BackendRejected as exc:
            if exc.status_code == 409:
                # 이미 완료·영구 실패된 신고는 다시 라우팅하지 않는다.
                return False
            if exc.status_code == 404:
                # 저장할 분석이 없어도 원본 이벤트는 DLQ에서 확인할 수 있게 한다.
                return True
            raise
        except (BackendUnavailable, BackendContractInvalid) as exc:
            # 상태 callback 장애가 원본 이벤트의 Retry/DLQ 이동을 막지 않게 한다.
            logger.warning(
                "moderation_failure_callback_failed report_id=%d failure_code=%s error_type=%s",
                report_id,
                failure_code,
                type(exc).__name__,
            )
            return True

    async def _publish_retry(
        self,
        message: Any,
        producer: Any,
        retry_index: int,
        failure_code: str,
    ) -> None:
        retry_at = self.now_ms() + self.settings.kafka_retry_delays_seconds[retry_index] * 1000
        headers = self._forward_headers(
            message,
            failure_code,
            extra=[(RETRY_AT_HEADER, str(retry_at).encode("ascii"))],
        )
        await producer.send_and_wait(
            f"{self.settings.kafka_topic}.retry-{retry_index}",
            value=message.value,
            key=message.key,
            headers=headers,
        )

    async def _publish_dlq(self, message: Any, producer: Any, failure_code: str) -> None:
        await producer.send_and_wait(
            f"{self.settings.kafka_topic}.dlq",
            value=message.value,
            key=message.key,
            headers=self._forward_headers(message, failure_code),
        )

    def _forward_headers(
        self,
        message: Any,
        failure_code: str,
        *,
        extra: list[tuple[str, bytes]] | None = None,
    ) -> list[tuple[str, bytes]]:
        internal = {RETRY_AT_HEADER, FAILURE_CODE_HEADER, ORIGINAL_TOPIC_HEADER}
        headers = [(key, value) for key, value in (message.headers or []) if key not in internal]
        headers.extend(
            [
                (FAILURE_CODE_HEADER, failure_code.encode("ascii")),
                (ORIGINAL_TOPIC_HEADER, self.settings.kafka_topic.encode("utf-8")),
            ]
        )
        headers.extend(extra or [])
        return headers

    def _classify_failure(self, exc: Exception) -> tuple[bool, str]:
        if isinstance(exc, BackendContractInvalid):
            return False, "BACKEND_CONTRACT_INVALID"
        if isinstance(exc, BackendRejected):
            return False, "BACKEND_REJECTED"
        if isinstance(exc, ValidationError):
            return False, "BACKEND_CONTRACT_INVALID"
        if isinstance(exc, TimeoutError):
            return True, "ANALYSIS_TIMEOUT"
        if isinstance(exc, BackendUnavailable):
            return True, "BACKEND_UNAVAILABLE"
        if isinstance(exc, ModelOutputError):
            return True, "MODEL_OUTPUT_INVALID"
        return True, "ANALYSIS_FAILED"

    def _last_header(self, headers: list[tuple[str, bytes]] | None, name: str) -> bytes | None:
        values = [value for key, value in (headers or []) if key == name]
        return values[-1] if values else None

    async def _wait_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass
