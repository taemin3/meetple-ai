# Meetple AI

> 신고 증거와 운영 정책을 연결해 관리자 검토용 분석을 만드는 FastAPI 서비스

Meetple 전체 구성과 저장소 링크는 [프로젝트 허브](https://github.com/taemin3/meetple)에서 확인할 수 있습니다.

## 현재 범위

이 저장소의 현재 제품·포트폴리오 범위는 **운영 정책 기반 신고 분석**입니다. 저장소에 남아 있는 자연어 모임 검색 구현은 현재 서비스 소개와 운영 대상에 포함하지 않습니다.

AI는 신고를 자동 처벌하지 않습니다. 증거와 정책을 바탕으로 위험도·우선순위·근거·추천 제재를 만들고, 실제 제재는 관리자가 Admin에서 확인한 뒤 Spring API를 통해 승인합니다.

## 처리 흐름

```text
Spring Outbox → Debezium → Kafka
                         │ reportId
                         ▼
FastAPI Consumer → Spring에서 신고 시점 증거 조회
                         │
                         ▼
LangGraph → 신고 요약 → 정책 검색 문장 생성 → 정책 RAG
                         │
                         ▼
OpenAI Structured Output → 증거·정책 ID와 인용문 검증
                         │
                         ▼
Spring callback → 관리자 검토 → 제재 승인
```

Kafka 이벤트에는 `reportId`만 포함합니다. AI 서비스는 Spring 내부 API에서 신고 시점 스냅샷을 조회하고, 분석 결과를 callback으로 저장합니다.

## 안전장치

- 사용자 JWT를 AI 서버나 모델에 전달하지 않고 서버 간 서비스 토큰을 사용합니다.
- 신고 원문, 정책 원문, 인증값과 공급자 오류 본문을 애플리케이션 로그에 남기지 않습니다.
- LLM 응답은 Pydantic Structured Output으로 파싱한 뒤 증거 ID·정책 ID·원문 인용을 코드로 다시 검증합니다.
- 관련 정책을 찾지 못하면 검색 문장을 최대 한 번 재작성하고, 근거 검증 실패 시 같은 후보 안에서 최대 한 번 보정합니다.
- 보정 단계는 최초 제재 수위를 높일 수 없으며 근거가 부족하면 `MANUAL_REVIEW` 또는 `DISMISS`로 제한합니다.
- Retry/DLQ와 Spring의 결과 멱등성 검증을 사용하지만 전달 의미는 at-least-once입니다.

## 기술 구성

| 구분 | 기술 |
| --- | --- |
| API | Python 3.12, FastAPI, Uvicorn |
| Workflow | LangGraph |
| Model | OpenAI Responses API, Structured Outputs, Embeddings |
| Event | aiokafka, Kafka Retry/DLQ |
| Validation | Pydantic, pytest, Ruff, 자체 평가 데이터 |
| Deployment | Docker, GitHub Actions OIDC, AWS ECR·ECS |

## 주요 경로

```text
src/meetple_ai/
  app.py                 FastAPI와 lifecycle
  moderation_graph.py    신고 분석·정책 검색·근거 검증 그래프
  moderation_worker.py   Kafka consumer, Retry/DLQ, callback
  backend.py             Spring 내부 API client
  model.py               OpenAI adapter
  settings.py            환경 설정
tests/                   외부 호출 없는 자동 테스트
evals/                   가상 데이터 기반 평가
```

## 로컬 실행

Python 3.12 이상을 사용합니다. PowerShell에서 저장소 루트 기준으로 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

`.env`에는 실제 값을 직접 입력하며 Git에 커밋하지 않습니다.

| 환경변수 | 역할 |
| --- | --- |
| `AI_SERVICE_TOKEN` | Spring과 공유하는 32자 이상의 서버 간 인증값 |
| `AI_OPENAI_API_KEY` | OpenAI API 키 |
| `AI_OPENAI_MODEL` | Structured Outputs를 지원하는 모델 ID |
| `AI_OPENAI_EMBEDDING_MODEL` | 정책 임베딩 모델. 현재 `text-embedding-3-small` |
| `AI_BACKEND_URL` | Spring 내부 API 주소 |
| `AI_KAFKA_CONSUMER_ENABLED` | 신고 분석 consumer 활성화. 기본 `false` |
| `AI_KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap 주소 |

Kafka와 Spring 연동 없이 내부 API만 확인할 때는 consumer를 비활성화한 상태로 실행할 수 있습니다.

```powershell
.venv\Scripts\python -m uvicorn meetple_ai.app:app --host 127.0.0.1 --port 8001
```

### 주요 endpoint

- `GET /healthz`: 프로세스 생존 여부
- `GET /readyz`: 모델 설정과 consumer 준비 여부
- `POST /v1/moderation/analyze`: 신고 분석 그래프 직접 검증
- `POST /v1/moderation/policies/embeddings/sync`: 운영 정책 임베딩 동기화

health endpoint를 제외한 요청은 `X-AI-Service-Token`을 요구합니다. 서비스는 공개 ingress가 아닌 Spring과 연결된 사설 네트워크에서 실행하는 것을 전제로 합니다.

## 검증

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python evals/run.py
```

기본 테스트와 평가는 외부 모델을 호출하지 않습니다. `evals/run.py --live`는 실제 OpenAI 호출과 비용이 발생하며 별도 품질 검증으로 구분합니다.

## 배포

`.github/workflows/ci.yml`은 PR과 `main` push에서 테스트를 실행합니다. staging 배포는 GitHub OIDC로 단기 AWS 자격 증명을 발급받고, commit SHA 이미지로 ECR과 ECS task revision을 갱신합니다. 자동 배포는 repository variable `AUTO_DEPLOY_ENABLED=true`일 때만 활성화됩니다.

## 운영 경계

- 기본 정책 점수 임계값과 후보 수는 초기 운영값이며 실제 정확도에 맞춘 최적값으로 측정된 것은 아닙니다.
- 자동 테스트 통과는 실제 모델 품질이나 신고 분류 정확도를 보장하지 않습니다.
- callback, Retry publish, offset commit은 하나의 Kafka 트랜잭션이 아니므로 중복 전달 가능성을 전제로 합니다.
- 위험한 제재는 AI가 실행하지 않으며 관리자 검토를 최종 경계로 유지합니다.

## 관련 저장소

- [Meetple Backend](https://github.com/taemin3/meetple-backend)
- [Meetple Admin](https://github.com/taemin3/meetple-admin)
- [Meetple App](https://github.com/taemin3/meetple-app)
