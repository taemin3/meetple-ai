from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")
    service_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    openai_embedding_model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    backend_url: str = "http://127.0.0.1:8080"
    mcp_url: str = "http://127.0.0.1:8001/mcp/"
    kafka_consumer_enabled: bool = False
    kafka_bootstrap_servers: str = "127.0.0.1:9092"
    kafka_consumer_group: str = "meetple-ai-report-analysis-v1"
    kafka_topic: str = "meetple.moderation.report-analysis.v1"
    kafka_retry_delays_seconds: tuple[int, int, int, int] = (5, 30, 120, 600)
    kafka_max_poll_interval_ms: int = 900_000

    @model_validator(mode="after")
    def validate_kafka(self):
        if not self.kafka_consumer_enabled:
            return self
        if not self.kafka_bootstrap_servers.strip():
            raise ValueError("Kafka bootstrap server가 필요합니다.")
        if not self.kafka_consumer_group.strip() or not self.kafka_topic.strip():
            raise ValueError("Kafka consumer group과 topic이 필요합니다.")
        if any(delay < 0 for delay in self.kafka_retry_delays_seconds):
            raise ValueError("Kafka retry 지연은 음수일 수 없습니다.")
        required_poll_interval = (max(self.kafka_retry_delays_seconds) + 60) * 1000
        if self.kafka_max_poll_interval_ms < required_poll_interval:
            raise ValueError("Kafka max poll interval이 retry 지연보다 짧습니다.")
        return self

    @property
    def ready(self) -> bool:
        return (
            len(self.service_token.get_secret_value()) >= 32
            and bool(self.openai_api_key.get_secret_value())
            and bool(self.openai_model.strip())
        )
