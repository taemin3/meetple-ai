from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

POLICY_CONTEXT_MAX_CHARS = 3000


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


ReportTargetType = Literal["MEMBER", "MEETING", "CHAT_MESSAGE"]
ReportReason = Literal[
    "SPAM",
    "ABUSE_OR_HARASSMENT",
    "INAPPROPRIATE_CONTENT",
    "FRAUD_OR_FALSE_INFORMATION",
    "OTHER",
]
PolicyType = Literal[
    "SPAM",
    "ABUSE_OR_HARASSMENT",
    "INAPPROPRIATE_CONTENT",
    "FRAUD_OR_FALSE_INFORMATION",
    "SAFETY",
    "GENERAL",
]
ModerationReportType = Literal[
    "SPAM",
    "ABUSE_OR_HARASSMENT",
    "INAPPROPRIATE_CONTENT",
    "FRAUD_OR_FALSE_INFORMATION",
    "SAFETY",
    "OTHER",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ModerationPriority = Literal["LOW", "NORMAL", "HIGH", "URGENT"]
RecommendedAction = Literal[
    "DISMISS",
    "WARNING",
    "SUSPEND_1_DAY",
    "SUSPEND_3_DAYS",
    "SUSPEND_7_DAYS",
    "PERMANENT_SUSPENSION",
    "FORCE_DELETE_MEETING",
    "MANUAL_REVIEW",
]


class ModerationEvidence(Contract):
    evidenceId: int = Field(gt=0)
    evidenceType: ReportTargetType
    content: str = Field(min_length=1, max_length=4000)


class ModerationAnalysisRequest(Contract):
    reportId: int = Field(gt=0)
    targetType: ReportTargetType
    reason: ReportReason
    description: str | None = Field(default=None, max_length=500)
    evidence: list[ModerationEvidence] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_evidence(self):
        evidence_ids = [item.evidenceId for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("증거 ID는 중복될 수 없습니다.")
        if sum(len(item.content) for item in self.evidence) > 12000:
            raise ValueError("신고 증거의 전체 길이가 너무 깁니다.")
        if self.reason == "OTHER" and not self.description:
            raise ValueError("기타 신고 사유에는 설명이 필요합니다.")
        return self


class PolicySearchPlan(Contract):
    summary: str = Field(min_length=1, max_length=500)
    keyword: str = Field(min_length=1, max_length=200)
    semanticQuery: str = Field(min_length=1, max_length=500)


class PolicyCandidate(Contract):
    policyId: int = Field(gt=0)
    policyChunkId: int = Field(gt=0)
    policyCode: str
    policyTitle: str
    policyType: PolicyType
    targetType: Literal["ALL", "MEMBER", "MEETING", "CHAT_MESSAGE"]
    policyVersion: int = Field(gt=0)
    clauseCode: str
    content: str = Field(min_length=1)
    contentHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveFrom: date
    effectiveTo: date | None
    keywordMatched: bool
    semanticDistance: float
    hybridScore: float


class PolicyCandidates(Contract):
    items: list[PolicyCandidate] = Field(max_length=20)
    hasMore: bool


class PolicyEmbeddingJob(Contract):
    policyId: int = Field(gt=0)
    policyChunkId: int = Field(gt=0)
    policyCode: str
    policyVersion: int = Field(gt=0)
    clauseCode: str
    content: str = Field(min_length=1)
    contentHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class PolicyEmbeddingJobs(Contract):
    items: list[PolicyEmbeddingJob] = Field(max_length=100)


class PolicyEmbeddingSyncRequest(Contract):
    limit: int = Field(default=50, ge=1, le=100)


class PolicyEmbeddingSyncResponse(Contract):
    requestedCount: int = Field(ge=0, le=100)
    embeddedCount: int = Field(ge=0, le=100)
    embeddingModel: Literal["text-embedding-3-small"]


class EvidenceGrounding(Contract):
    evidenceId: int = Field(gt=0)
    evidenceQuote: str = Field(min_length=1, max_length=500)


class PolicyGrounding(Contract):
    policyId: int = Field(gt=0)
    policyChunkId: int = Field(gt=0)
    policyQuote: str = Field(min_length=1, max_length=500)


class ModerationDecision(Contract):
    reportType: ModerationReportType
    riskLevel: RiskLevel
    priority: ModerationPriority
    rationale: str = Field(min_length=1, max_length=1000)
    evidence: list[EvidenceGrounding] = Field(min_length=1, max_length=10)
    policies: list[PolicyGrounding] = Field(min_length=1, max_length=10)
    confidence: float = Field(ge=0, le=1)
    recommendedAction: RecommendedAction


class ModerationAnalysisResponse(Contract):
    reportId: int = Field(gt=0)
    reportType: ModerationReportType
    riskLevel: RiskLevel
    priority: ModerationPriority
    summary: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)
    evidenceIds: list[int] = Field(min_length=1, max_length=10)
    policyIds: list[int] = Field(min_length=1, max_length=10)
    confidence: float = Field(ge=0, le=1)
    recommendedAction: RecommendedAction
