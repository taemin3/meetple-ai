from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")
    service_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    openai_embedding_model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    backend_url: str = "http://127.0.0.1:8080"
    mcp_url: str = "http://127.0.0.1:8001/mcp/"

    @property
    def ready(self) -> bool:
        return (
            len(self.service_token.get_secret_value()) >= 32
            and bool(self.openai_api_key.get_secret_value())
            and bool(self.openai_model.strip())
        )
