from datetime import datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class SearchRequest(Contract):
    query: str = Field(min_length=1, max_length=1000)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    radiusMeters: int = Field(ge=100, le=50000)
    referenceTime: datetime

    @model_validator(mode="after")
    def validate_location_and_time(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("위도와 경도는 함께 전달해야 합니다.")
        if self.referenceTime.tzinfo is not None:
            raise ValueError("referenceTime은 Spring이 설정한 한국 현지 시각이어야 합니다.")
        return self


class Intent(Contract):
    keyword: str = Field(max_length=100)
    semanticQuery: str | None = Field(min_length=1, max_length=500)
    category: str | None
    dateMode: Literal["any", "today", "tomorrow", "this_weekend", "next_weekend", "range"]
    startDate: str | None
    endDate: str | None
    timeMode: Literal["any", "morning", "afternoon", "evening", "range"]
    startTime: str | None
    endTime: str | None
    radiusMeters: int | None
    unsupportedReason: str | None

    @field_validator("semanticQuery", mode="before")
    @classmethod
    def normalize_blank_semantic_query(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class Filters(Contract):
    keyword: str = Field(max_length=100)
    category: str | None
    startsAt: datetime
    endsBefore: datetime
    startsAtTime: time | None
    endsBeforeTime: time | None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radiusMeters: int = Field(ge=100, le=50000)


class Candidate(Contract):
    id: int = Field(gt=0)
    title: str
    description: str
    categoryName: str
    locationName: str
    scheduledAt: datetime
    endsAt: datetime | None
    capacity: int
    currentPeople: int
    distanceMeters: float = Field(ge=0)


class Candidates(Contract):
    items: list[Candidate] = Field(max_length=20)
    hasMore: bool


class Recommendation(Contract):
    meetingId: int = Field(gt=0)
    evidenceQuote: str = Field(min_length=1, max_length=500)


class Selection(Contract):
    recommendations: list[Recommendation] = Field(max_length=5)


class SearchResponse(Contract):
    status: Literal["COMPLETED", "NO_RESULTS", "INPUT_REQUIRED", "UNSUPPORTED"]
    message: str
    filters: Filters | None
    recommendations: list[Recommendation]
    retrievalMode: Literal["keyword"] = "keyword"


class MeetingEmbeddingRequest(Contract):
    document: str = Field(min_length=1, max_length=3000)


class MeetingEmbeddingResponse(Contract):
    embeddingModel: str = Field(min_length=1, max_length=100)
    embedding: list[float] = Field(min_length=1536, max_length=1536)
