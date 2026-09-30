from datetime import datetime

import pytest

from meetple_ai.contracts import Candidate, Candidates, Intent, SearchRequest, Selection


@pytest.fixture
def request_data():
    return SearchRequest(
        query="이번 주말 초보 러닝 모임",
        latitude=37.5,
        longitude=127,
        radiusMeters=3000,
        referenceTime=datetime(2026, 9, 29, 12),
    )


@pytest.fixture
def intent():
    return Intent(
        keyword="러닝",
        category="운동",
        dateMode="this_weekend",
        startDate=None,
        endDate=None,
        timeMode="any",
        startTime=None,
        endTime=None,
        radiusMeters=None,
        unsupportedReason=None,
    )


@pytest.fixture
def candidate():
    return Candidate(
        id=10,
        title="주말 러닝",
        description="처음 달리는 분 환영",
        categoryName="운동",
        locationName="공원",
        scheduledAt=datetime(2026, 10, 3, 15),
        endsAt=None,
        capacity=6,
        currentPeople=2,
        distanceMeters=300,
    )


class FakeModel:
    def __init__(self, intent, recommendations=None):
        self.intent = intent
        self.selection = Selection(recommendations=recommendations or [])
        self.calls = []

    async def interpret(self, request, categories):
        self.calls.append("interpret")
        return self.intent

    async def select(self, request, candidates):
        self.calls.append("select")
        return self.selection


class FakeTools:
    def __init__(self, items=(), has_more=False):
        self.items, self.has_more = list(items), has_more
        self.calls = []

    async def categories(self):
        self.calls.append("categories")
        return ["운동", "취미", "스터디"]

    async def search(self, filters):
        self.calls.append(filters)
        return Candidates(items=self.items, hasMore=self.has_more)
