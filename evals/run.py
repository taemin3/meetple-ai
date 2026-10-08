"""Synthetic-data evaluation. --live explicitly enables paid OpenAI calls."""

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import monotonic

from openai import AsyncOpenAI

from meetple_ai.contracts import (
    Candidate,
    Candidates,
    ModerationAnalysisRequest,
    PolicyCandidate,
    PolicyCandidates,
    SearchRequest,
)
from meetple_ai.graph import build_graph
from meetple_ai.model import OpenAISearchModel
from meetple_ai.moderation_graph import build_moderation_graph
from meetple_ai.settings import Settings

ROOT = Path(__file__).parent
CATEGORIES = ["운동", "스터디", "취미", "친목", "여행", "맛집", "비즈니스", "반려동물"]


def load_meeting_data():
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


def load_moderation_data():
    cases = json.loads((ROOT / "moderation_cases.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "moderation_policies.json").read_text(encoding="utf-8"))
    policies = [PolicyCandidate.model_validate(item) for item in raw]
    assert len({case["id"] for case in cases}) == len(cases)
    policy_ids = {policy.policyId for policy in policies}
    for case in cases:
        make_moderation_request(case)
        expected = case["expected"]
        assert set(expected["requiredPolicyIds"]) <= set(expected["allowedPolicyIds"])
        assert set(expected["allowedPolicyIds"]) <= policy_ids
        assert expected["reportTypes"] and expected["riskLevels"] and expected["actions"]
        recovery = case.get("recovery")
        if recovery:
            assert recovery["type"] in {"SEARCH_REWRITE", "GROUNDING_REPAIR"}
            if recovery["type"] == "SEARCH_REWRITE":
                assert recovery["requiredTerms"]
    return cases, policies


def make_request(case):
    located = case.get("location", True)
    return SearchRequest(
        query=case["query"],
        latitude=37.5 if located else None,
        longitude=127 if located else None,
        radiusMeters=3000,
        referenceTime=case.get("referenceTime", "2026-09-30T12:00:00"),
    )


def make_moderation_request(case):
    return ModerationAnalysisRequest.model_validate(case["request"])


class FixtureTools:
    """SQL-equivalent fixture filtering for model evaluation, not DB/MCP integration proof."""

    def __init__(self, candidates, embeddings=None):
        self.candidates = candidates
        self.embeddings = embeddings or {}

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
        ]
        if query_embedding is None:
            items = [c for c in items if f.keyword.lower() in (c.title + " " + c.description).lower()]
            items.sort(key=lambda c: (c.distanceMeters, c.scheduledAt, c.id))
        else:
            if set(c.id for c in items) - self.embeddings.keys():
                raise ValueError("평가 후보 임베딩이 없습니다.")
            items.sort(
                key=lambda c: (
                    cosine_distance(query_embedding, self.embeddings[c.id]),
                    c.distanceMeters,
                    c.scheduledAt,
                    c.id,
                )
            )
        return Candidates(items=items[:20], hasMore=len(items) > 20)


class ModerationFixtureTools:
    """Target filtering and deliberate recovery injection for prompt evaluation."""

    def __init__(self, policies, recovery=None):
        self.policies = policies
        self.recovery = recovery or {}
        self.search_calls = 0
        self.first_query = None

    async def search_policies(self, request, plan, query_embedding, limit):
        self.search_calls += 1
        items = [policy for policy in self.policies if policy.targetType in ("ALL", request.targetType)]
        if self.recovery.get("type") == "SEARCH_REWRITE":
            current_query = (plan.keyword, plan.semanticQuery)
            if self.search_calls == 1:
                self.first_query = current_query
                items = [policy.model_copy(update={"hybridScore": -1.0}) for policy in items]
            else:
                combined_query = f"{plan.keyword} {plan.semanticQuery}".lower()
                has_required_term = any(
                    term.lower() in combined_query for term in self.recovery["requiredTerms"]
                )
                if current_query == self.first_query or not has_required_term:
                    items = [policy.model_copy(update={"hybridScore": -1.0}) for policy in items]
        return PolicyCandidates(items=items[:limit], hasMore=len(items) > limit)


class ModerationEvaluationModel:
    """Delegates to the live model while forcing the requested recovery branch once."""

    def __init__(self, model, recovery=None):
        self.model = model
        self.recovery = recovery or {}
        self.refine_calls = 0
        self.repair_calls = 0
        self.grounding_corrupted = False

    async def prepare_moderation(self, request):
        return await self.model.prepare_moderation(request)

    async def refine_moderation_search(self, request, previous_plan):
        self.refine_calls += 1
        return await self.model.refine_moderation_search(request, previous_plan)

    async def embed(self, semantic_query):
        return await self.model.embed(semantic_query)

    async def analyze_moderation(self, request, plan, policies):
        decision = await self.model.analyze_moderation(request, plan, policies)
        if self.recovery.get("type") != "GROUNDING_REPAIR" or self.grounding_corrupted:
            return decision
        self.grounding_corrupted = True
        invalid_evidence = decision.evidence[0].model_copy(update={"evidenceId": 2_147_483_647})
        return decision.model_copy(update={"evidence": [invalid_evidence, *decision.evidence[1:]]})

    async def repair_moderation_decision(
        self,
        request,
        plan,
        policies,
        previous_decision,
        validation_error,
    ):
        self.repair_calls += 1
        return await self.model.repair_moderation_decision(
            request,
            plan,
            policies,
            previous_decision,
            validation_error,
        )


def grade_recovery(case, model, tools):
    recovery_type = case.get("recovery", {}).get("type")
    if recovery_type == "SEARCH_REWRITE":
        return model.refine_calls == 1 and tools.search_calls == 2
    if recovery_type == "GROUNDING_REPAIR":
        return model.grounding_corrupted and model.repair_calls == 1
    return model.refine_calls == 0 and model.repair_calls == 0


def cosine_distance(left, right):
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise ValueError("평가 임베딩의 크기가 0입니다.")
    return 1 - sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def candidate_document(candidate):
    return (
        f"제목: {candidate.title}\n카테고리: {candidate.categoryName}\n"
        f"장소: {candidate.locationName}\n소개: {candidate.description}"
    )


def grade(case, response):
    actual = {r.meetingId for r in response.recommendations}
    result = {
        "status": response.status == case["status"],
        "ids": not case.get("checkIds", True) or actual == set(case["ids"]),
    }
    filters = response.filters.model_dump(mode="json") if response.filters else {}
    result["filters"] = all(filters.get(k) == v for k, v in case.get("filters", {}).items())
    return result


def grade_moderation(case, response):
    expected = case["expected"]
    policy_ids = set(response.policyIds)
    return {
        "reportType": response.reportType in expected["reportTypes"],
        "riskLevel": response.riskLevel in expected["riskLevels"],
        "action": response.recommendedAction in expected["actions"],
        "evidenceIds": set(response.evidenceIds) == set(expected["evidenceIds"]),
        "requiredPolicies": set(expected["requiredPolicyIds"]) <= policy_ids,
        "allowedPolicies": policy_ids <= set(expected["allowedPolicyIds"]),
    }


async def evaluate_meetings(cases, candidates):
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
        candidate_embeddings = await model.embed_many(
            [candidate_document(candidate) for candidate in candidates]
        )
        tools = FixtureTools(
            candidates,
            {candidate.id: embedding for candidate, embedding in zip(candidates, candidate_embeddings)},
        )
        for case in cases:
            started = monotonic()
            row = {"id": case["id"]}
            try:
                async with asyncio.timeout(35):
                    result = await build_graph(model, tools).ainvoke(
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
        "scope": "synthetic fixtures with in-memory cosine ranking; no Spring/MCP/DB",
        "count": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "meanMs": round(mean(durations)),
        "p50Ms": durations[(len(rows) - 1) // 2],
        "p95Ms": durations[max(0, (95 * len(rows) + 99) // 100 - 1)],
        "cases": rows,
    }
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-meeting.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{len(rows)} passed; {output}")
    return 0 if report["passed"] == len(rows) else 1


async def evaluate_moderation(cases, policies):
    settings = Settings()
    if not settings.openai_api_key.get_secret_value() or not settings.openai_model:
        raise SystemExit("AI_OPENAI_API_KEY와 AI_OPENAI_MODEL을 설정해주세요.")
    rows = []
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(), timeout=12, max_retries=0
    ) as client:
        live_model = OpenAISearchModel(client, settings.openai_model, settings.openai_embedding_model)
        for case in cases:
            started = monotonic()
            row = {"id": case["id"]}
            model = ModerationEvaluationModel(live_model, case.get("recovery"))
            tools = ModerationFixtureTools(policies, case.get("recovery"))
            try:
                async with asyncio.timeout(75):
                    result = await build_moderation_graph(
                        model,
                        tools,
                        min_policy_hybrid_score=settings.moderation_policy_min_hybrid_score,
                        policy_result_limit=settings.moderation_policy_result_limit,
                    ).ainvoke({"request": make_moderation_request(case)}, {"recursion_limit": 20})
                row["checks"] = grade_moderation(case, result["response"])
                row["checks"]["recoveryBranch"] = grade_recovery(case, model, tools)
                row["passed"] = all(row["checks"].values())
                row["actual"] = result["response"].model_dump(mode="json")
            except Exception as exc:
                row.update(passed=False, errorType=type(exc).__name__)
            row["durationMs"] = round((monotonic() - started) * 1000)
            rows.append(row)
            print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'}")
    durations = sorted(row["durationMs"] for row in rows)
    report = {
        "model": settings.openai_model,
        "embeddingModel": settings.openai_embedding_model,
        "scope": (
            "synthetic reports and policies; in-memory target filter with deliberate recovery "
            "injections; no Spring/DB"
        ),
        "count": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "meanMs": round(mean(durations)),
        "p50Ms": durations[(len(rows) - 1) // 2],
        "p95Ms": durations[max(0, (95 * len(rows) + 99) // 100 - 1)],
        "cases": rows,
    }
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-moderation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{len(rows)} passed; {output}")
    return 0 if report["passed"] == len(rows) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="유료 OpenAI 호출 활성화",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "meeting", "moderation"),
        default="all",
        help="평가 묶음 선택 (기본: all)",
    )
    parser.add_argument("--limit", type=int, help="실행할 평가 문항 수 (기본: 전체)")
    parser.add_argument("--case", help="실행할 평가 문항 ID 하나")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    meeting_cases, candidates = load_meeting_data()
    moderation_cases, policies = load_moderation_data()
    if not args.live:
        print(
            f"Validated {len(meeting_cases)} meeting cases, {len(candidates)} meeting candidates, "
            f"{len(moderation_cases)} moderation cases, and {len(policies)} policies. No API calls."
        )
        return 0

    selected_meetings = []
    selected_moderation = []
    if args.suite in ("all", "meeting"):
        selected_meetings = (
            [case for case in meeting_cases if case["id"] == args.case]
            if args.case
            else meeting_cases[: args.limit]
        )
    if args.suite in ("all", "moderation"):
        selected_moderation = (
            [case for case in moderation_cases if case["id"] == args.case]
            if args.case
            else moderation_cases[: args.limit]
        )
    if not selected_meetings and not selected_moderation:
        parser.error(f"unknown case: {args.case}")

    async def run_selected():
        results = []
        if selected_meetings:
            results.append(await evaluate_meetings(selected_meetings, candidates))
        if selected_moderation:
            results.append(await evaluate_moderation(selected_moderation, policies))
        return max(results)

    return asyncio.run(run_selected())


if __name__ == "__main__":
    raise SystemExit(main())
