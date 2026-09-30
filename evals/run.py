"""Synthetic-data evaluation. --live explicitly enables paid OpenAI calls."""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import monotonic

from openai import AsyncOpenAI

from meetple_ai.contracts import Candidate, Candidates, SearchRequest
from meetple_ai.graph import build_graph
from meetple_ai.model import OpenAISearchModel
from meetple_ai.settings import Settings

ROOT = Path(__file__).parent
CATEGORIES = ["운동", "스터디", "취미", "친목", "여행", "맛집", "비즈니스", "반려동물"]


def load_data():
    cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    candidates = [
        Candidate(locationName="가상 공원", endsAt=None, capacity=10, currentPeople=2, **r) for r in raw
    ]
    assert len({c["id"] for c in cases}) == len(cases)
    ids = {c.id for c in candidates}
    for case in cases:
        make_request(case)
        assert set(case["ids"]) <= ids
        assert case["status"] in ("COMPLETED", "NO_RESULTS", "INPUT_REQUIRED", "UNSUPPORTED")
    return cases, candidates


def make_request(case):
    located = case.get("location", True)
    return SearchRequest(
        query=case["query"],
        latitude=37.5 if located else None,
        longitude=127 if located else None,
        radiusMeters=3000,
        referenceTime=case.get("referenceTime", "2026-09-30T12:00:00"),
    )


class FixtureTools:
    """SQL-equivalent fixture filtering for model evaluation, not DB/MCP integration proof."""

    def __init__(self, candidates):
        self.candidates = candidates

    async def categories(self):
        return CATEGORIES

    async def search(self, f, query_embedding):
        items = [
            c
            for c in self.candidates
            if (not f.category or c.categoryName == f.category)
            and f.startsAt <= c.scheduledAt < f.endsBefore
            and (not f.startsAtTime or c.scheduledAt.time() >= f.startsAtTime)
            and (not f.endsBeforeTime or c.scheduledAt.time() < f.endsBeforeTime)
            and c.distanceMeters <= f.radiusMeters
            and f.keyword.lower() in (c.title + " " + c.description).lower()
        ]
        items.sort(key=lambda c: (c.distanceMeters, c.scheduledAt, c.id))
        return Candidates(items=items[:20], hasMore=len(items) > 20)


def grade(case, response):
    actual = {r.meetingId for r in response.recommendations}
    result = {
        "status": response.status == case["status"],
        "ids": not case.get("checkIds", True) or actual == set(case["ids"]),
    }
    filters = response.filters.model_dump(mode="json") if response.filters else {}
    result["filters"] = all(filters.get(k) == v for k, v in case.get("filters", {}).items())
    return result


async def evaluate(cases, candidates):
    settings = Settings()
    if (
        not settings.openai_api_key.get_secret_value()
        or not settings.openai_model
        or not settings.openai_embedding_model
    ):
        raise SystemExit("AI_OPENAI_API_KEY, AI_OPENAI_MODEL, AI_OPENAI_EMBEDDING_MODEL을 설정해주세요.")
    rows = []
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(), timeout=12, max_retries=0
    ) as client:
        model = OpenAISearchModel(client, settings.openai_model, settings.openai_embedding_model)
        for case in cases:
            started = monotonic()
            row = {"id": case["id"]}
            try:
                async with asyncio.timeout(35):
                    result = await build_graph(model, FixtureTools(candidates)).ainvoke(
                        {"request": make_request(case)}, {"recursion_limit": 10}
                    )
                row["checks"] = grade(case, result["response"])
                row["passed"] = all(row["checks"].values())
                row["actual"] = result["response"].model_dump(mode="json")
            except Exception as exc:
                row.update(passed=False, errorType=type(exc).__name__)
            row["durationMs"] = round((monotonic() - started) * 1000)
            rows.append(row)
            print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'}")
    durations = sorted(r["durationMs"] for r in rows)
    report = {
        "model": settings.openai_model,
        "embeddingModel": settings.openai_embedding_model,
        "scope": "synthetic fixtures; no Spring/MCP/DB",
        "count": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "meanMs": round(mean(durations)),
        "p50Ms": durations[(len(rows) - 1) // 2],
        "p95Ms": durations[max(0, (95 * len(rows) + 99) // 100 - 1)],
        "cases": rows,
    }
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{len(rows)} passed; {output}")
    return 0 if report["passed"] == len(rows) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="유료 OpenAI 호출 활성화 (질문당 최대 3회)")
    parser.add_argument("--limit", type=int, help="실행할 평가 문항 수 (기본: 전체)")
    parser.add_argument("--case", help="실행할 평가 문항 ID 하나")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cases, candidates = load_data()
    if not args.live:
        print(f"Validated {len(cases)} cases and {len(candidates)} synthetic candidates. No API calls.")
        return 0
    selected = [case for case in cases if case["id"] == args.case] if args.case else cases[: args.limit]
    if not selected:
        parser.error(f"unknown case: {args.case}")
    return asyncio.run(evaluate(selected, candidates))


if __name__ == "__main__":
    raise SystemExit(main())
