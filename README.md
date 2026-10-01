# Meetple AI 서비스

자연어 모임 검색과 운영 정책에 근거한 신고 분석을 제공하는 Python AI 서버다. 코드·의존성·테스트·Docker 이미지·CI를 이 저장소에서 독립적으로 관리한다.

## 저장소와 폴더

권장 로컬 배치는 다음과 같다. 각 폴더는 별도의 Git 저장소다.

```text
C:\project\meetple\
  app\        Flutter 앱
  backend\    Spring API 서버
  ai\         Python AI 서버 (이 저장소)
```

Spring은 `backend` 폴더에서 실행한다. AI 응답의 `INPUT_REQUIRED`·`UNSUPPORTED` 상태와 시간 필드는
[meetple-backend PR #96](https://github.com/taemin3/meetple-backend/pull/96)에서 `main`에 반영됐다.
AI 서버는 이 계약을 포함한 백엔드 버전과 함께 배포한다. 서버 연결은 폴더 상대 경로가 아닌 HTTP 환경변수로 설정한다.

```text
ai/
  src/meetple_ai/       FastAPI, LangGraph, MCP, OpenAI 어댑터
  tests/               외부 호출 없는 자동 테스트
  evals/               가상 데이터와 모델 평가 실행기
  .github/workflows/   AI 전용 CI
  .env.example         환경변수 예시
  Dockerfile           AI 서버 이미지
  pyproject.toml       Python 패키지 설정
  requirements.lock    검증한 의존성 버전
```

Spring의 로그인·권한 검증, 모임 DB 조회와 최종 추천 재검증은 [meetple-backend](https://github.com/taemin3/meetple-backend)에 남아 있다. 초기 코드는 백엔드 커밋 `f8dcc7e`의 `ai-service/`에서 분리했다.

## 처리 흐름과 기술의 역할

```text
클라이언트 → Spring 로그인 검증 → Python FastAPI
  → LangGraph: 위치 확인 → 조건 추출 → 날짜/반경 검증 → 선택적 질문 임베딩
  → MCP: search_meetings → Spring 내부 API → PostgreSQL/PostGIS/pgvector
  → OpenAI: 후보 선택 + 원문 인용 → Python 검증
  → Spring: 차단·모집 상태와 원문 재검증 → 클라이언트
```

- **FastAPI**: 내부 검색 요청, 상태 확인, MCP HTTP 경로 제공.
- **LangGraph**: 검색 단계와 조건부 분기 관리. 빈 결과면 두 번째 모델 호출 생략.
- **OpenAI Responses API / Structured Outputs**: 조건과 추천 결과를 Pydantic 스키마로 파싱. 자동 재시도 없음.
- **OpenAI Embeddings API**: 의미 조건이 있는 `semanticQuery`를 1536차원 질문 벡터로 변환한다. 광범위한 검색은 호출을 생략한다.
- **MCP Python SDK**: `list_categories`, `search_meetings` 읽기 도구 제공. 실제 Streamable HTTP 프로토콜로 호출한다.
- **PostgreSQL/PostGIS**: 날짜·카테고리·반경·모집 여부와 차단 관계로 후보를 제한한다.
- **근거 검증**: 추천 ID가 실제 후보에 있고 인용문이 제목/본문의 연속된 원문인지 Python과 Spring에서 확인한다. 원문 검증만으로 의미적 적합성까지 보장하지는 않는다.

현재 응답의 `retrievalMode=keyword`는 유지한다. AI 서버는 질문 임베딩과 `AI_OPENAI_EMBEDDING_MODEL` 식별자를 Spring 내부 검색 API로 함께 전달한다. Spring은 같은 모델로 저장된 모임 벡터만 pgvector 의미 검색에 사용한다. 모임 임베딩 갱신·백필, 자유로운 에이전트 도구 선택, 일정 충돌 확인, 채팅 요약과 Flutter 화면은 후속 범위다.

## 검색 정책

- 요청 시점은 Spring이 `Asia/Seoul` 기준으로 설정한다. 클라이언트가 날짜 기준을 주입하지 않는다.
- 날짜 미지정 시 오늘부터 30일 범위, 이미 시작한 모임 제외. 주말은 월요일 기준 토·일이며 일요일의 '이번 주말'은 당일 남은 시간이다.
- 날짜 범위의 끝 날짜는 포함한다. DB에는 다음 날 00시 미만 조건으로 전달한다.
- 반경 100~50,000m. 모델은 앱이 전달한 반경보다 넓힐 수 없고 검색 중심을 변경할 수 없다.
- 가까운 순으로 최대 20개 후보를 조회하고 최대 5개를 선택한다. 후보가 더 있으면 응답 메시지로 알린다.
- 오전은 06:00~12:00, 오후는 12:00~18:00, 저녁은 18:00 이후로 해석한다. 명확한 시각은 1시간 범위로 검색하며 `이후`·`이전`은 한쪽 경계로 적용한다.
- 시작 시간이 종료 시간보다 늦은 시간 범위는 자정을 지나는 구간으로 해석한다. 예를 들어 `23:30`~`00:30`은 두 시각 사이의 자정 전후 모임을 검색한다.
- 일반적인 선호 표현은 가장 자연스럽고 포괄적인 의미로 해석해 바로 검색한다. `초보자도 가능한`은 초보자를 환영하는 모임으로 해석한다.
- 좌표 없음은 `INPUT_REQUIRED`, 다른 지역·생성/참여 요청·일정 충돌 조건은 `UNSUPPORTED`를 반환한다. 지역명 → 좌표 변환과 대화 상태 저장은 아직 없다.
- 초보자 가능 여부는 소개글의 근거를 바탕으로 선택한다. 해당 필드가 DB에 별도로 있는 것은 아니다.

## 로컬 실행

AI 서버에는 Python 3.12+가 필요하다. 전체 연동에는 Java 21과 기존 Spring DB/Redis 환경도 필요하다. PowerShell에서 **이 저장소 루트(`C:\project\meetple\ai`)** 기준:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

`requirements.lock`은 검증 시점의 런타임·테스트 의존성을 고정한다. Windows 전용 `pywin32`에는 플랫폼 조건이 있다.

이 저장소 루트의 `.env`에 설정한다. 파일은 Git에서 제외된다. 이미 `.env`가 있으면 예시 파일로 덮어쓰지 않는다.

| 변수 | 값/역할 |
| --- | --- |
| `AI_SERVICE_TOKEN` | Spring과 공유하는 임의의 32자 이상 키 |
| `AI_OPENAI_API_KEY` | OpenAI API 키 |
| `AI_OPENAI_MODEL` | 계정에서 사용 가능한 Responses + Structured Outputs 지원 모델 ID |
| `AI_OPENAI_EMBEDDING_MODEL` | `vector(1536)`과 맞는 고정 모델 `text-embedding-3-small` |
| `AI_BACKEND_URL` | 기본 `http://127.0.0.1:8080` |
| `AI_MCP_URL` | 기본 `http://127.0.0.1:8001/mcp/` |

생성 모델 ID에는 기본값이 없으며 환경변수로 지정한다. 임베딩 모델은 Spring의 저장 벡터와 같은 `text-embedding-3-small`만 허용한다. 계정의 모델 접근 권한과 실제 응답 품질은 직접 호출해 확인해야 한다.

```powershell
.venv\Scripts\python -m uvicorn meetple_ai.app:app --host 127.0.0.1 --port 8001
```

Spring 실행 환경에도 다음 값을 넣는다. Python `.env`는 Spring이 자동으로 읽지 않는다.

| 변수 | 값/역할 |
| --- | --- |
| `AI_SEARCH_ENABLED` | 기본 `false`; 로컬 연결 시 `true` |
| `AI_SEARCH_BASE_URL` | `http://127.0.0.1:8001` |
| `AI_SEARCH_SERVICE_TOKEN` | Python의 `AI_SERVICE_TOKEN`과 동일 |
| `AI_SEARCH_CAPABILITY_SECRET` | 공유 키와 다른 32자 이상 임의 키. **Spring에만 설정** |
| `AI_SEARCH_TIMEOUT` | 기본 `45s`, 최대 `60s` |

`GET http://127.0.0.1:8001/healthz`는 프로세스 상태, `/readyz`는 생성 모델과 임베딩 모델 설정 유무만 확인한다. 실제 OpenAI 연결·잔액·모델 권한을 검사하는 프로브가 아니다.

이 저장소 루트에서 컨테이너 빌드: `docker build -t meetple-ai .`. 컨테이너 실행 시 `AI_BACKEND_URL`에는 Spring에 접근 가능한 사설 주소를 지정한다. Docker Compose/ECS 배포 설정은 이번 범위에 포함하지 않는다.

## 앱용 API 계약

`POST /api/v1/meetings/ai-search`, 기존 `Authorization: Bearer <accessToken>` 필요.

```json
{
  "query": "이번 주말 초보자도 가능한 러닝 모임",
  "latitude": 37.5,
  "longitude": 127.0,
  "radiusMeters": 3000
}
```

응답은 기존 `ApiResponse`로 감싸며, 아래는 `data` 예시다. 모임 ID와 인용문은 설명용이다.

```json
{
  "status": "COMPLETED",
  "message": "검색 조건과 소개글을 확인한 모임입니다.",
  "filters": {
    "keyword": "러닝",
    "category": "운동",
    "startsAt": "2026-10-03T00:00:00",
    "endsBefore": "2026-10-05T00:00:00",
    "startsAtTime": "18:00:00",
    "endsBeforeTime": null,
    "latitude": 37.5,
    "longitude": 127.0,
    "radiusMeters": 3000
  },
  "recommendations": [{"meetingId": 10, "evidenceQuote": "처음 달리는 분 환영"}],
  "retrievalMode": "keyword"
}
```

- `COMPLETED`: 추천 결과 있음. 모임 카드는 기존 모임 상세 API로 조회 가능하며 현재 응답에 카드 전체 필드는 없다.
- `NO_RESULTS`: 조회 또는 선호 조건 확인 결과 없음. 추천 목록은 비어 있다.
- `INPUT_REQUIRED`: 위치 또는 유효한 검색 조건이 필요하다. `filters=null`, 추천 목록은 비어 있다.
- `UNSUPPORTED`: 지역명 검색, 일정 충돌 확인, 생성·참여처럼 현재 검색 범위를 벗어난 요청이다. `filters=null`, 추천 목록은 비어 있다.
- HTTP 400: 잘못된 입력, 401: 로그인 필요, 502/코드 15201: AI 결과 재검증 실패, 503/코드 15301: 기능 비활성·AI 장애/시간 초과.

## 인증과 운영 경계

- 사용자 JWT는 Python이나 모델에 전달하지 않는다. Spring이 회원 ID·용도·90초 만료를 HMAC으로 서명한 검색 전용 권한을 만든다.
- 내부 API는 공유 서비스 키와 서명을 모두 검증한다. 회원 ID를 모델/도구 인자로 받지 않는다. 서명 키는 Python에 전달하지 않는다.
- Spring의 `/internal/ai/search/categories`, `/internal/ai/search/meetings`만 JWT 검사 대신 위 인증을 사용한다. Python, MCP와 내부 경로는 사설 네트워크에서 연결하고 공개 ingress에서는 차단해야 한다.
- 로그아웃 직전에 발급된 검색 권한은 최대 90초 유효할 수 있다. 조회에서는 탈퇴 회원과 차단한 모임장을 제외하며 최종 추천 직전에 다시 조회한다.
- DB 조회 중에만 DB 연결을 사용한다. LLM 응답을 기다리는 동안 Spring 트랜잭션을 유지하지 않는다.
- Python 프로세스당 동시 검색 4개, 대기 포함 전체 35초, 모델 호출당 12초. 사용자별/분산 요청 제한과 비용 한도는 아직 없으므로 운영 활성화 전에 추가해야 한다.
- OpenAI에는 질문·기준 시각·카테고리·위치 제공 여부와 후보 제목/설명/일시/거리만 전송한다. 의미 조건이 있으면 모델이 만든 `semanticQuery`도 임베딩 API에 전달한다. 실제 좌표·회원 정보·인증 헤더는 모델 입력에서 제외한다. 설명은 후보당 1,800자로 제한한다.
- 모델 호출에는 `store=false`를 지정한다. 외부 공급자의 모든 데이터 보관 정책을 제어한다는 뜻은 아니다. 실제 사용자 데이터로 출시하기 전 개인정보 안내를 검토해야 한다.

## 신고 분석 내부 API

`POST /v1/moderation/analyze`는 Spring 전용 서비스 키를 요구한다. 사용자 JWT나 모임 검색용 capability는 전달하지 않는다.

```json
{
  "reportId": 77,
  "targetType": "CHAT_MESSAGE",
  "reason": "ABUSE_OR_HARASSMENT",
  "description": null,
  "evidence": [
    {
      "evidenceId": 501,
      "evidenceType": "CHAT_MESSAGE",
      "content": "검증할 신고 증거 원문"
    }
  ]
}
```

처리 흐름은 다음과 같다.

```text
입력 검증 → 신고 요약·정책 검색 문장 생성 → text-embedding-3-small 임베딩
→ Spring 운영 정책 검색 API → 구조화된 신고 분류 → 증거·정책 ID와 원문 인용 검증
→ 검증된 ID만 분석 결과로 반환
```

초기 LLM 판단은 정책 유형의 하드 필터로 사용하지 않으며 대상 유형에 맞는 정책 전체에서 근거를 찾는다. 응답에는 신고 유형, 위험도, 우선순위, 요약, 판단 근거, 검증된 증거·정책 ID, 확신도와 관리자용 추천 제재가 포함된다. 위험도와 맞지 않거나 신고 대상에 적용할 수 없는 제재는 `MANUAL_REVIEW`로 제한한다. LLM에는 조회나 제재 도구를 제공하지 않으며, 추천만 생성한다. 실제 결과 저장, 자동 경고 조건 평가, 정지·삭제 같은 제재 실행은 Spring의 후속 단계다.

현재 엔드포인트는 HTTP 계약과 분석 그래프를 검증하기 위한 내부 처리 경계다. Kafka 이벤트 소비, 신고 문맥 조회, Spring 결과 콜백과 Retry/DLQ는 backend 연동 PR에서 구현한다.

`POST /v1/moderation/policies/embeddings/sync`는 누락되거나 정책 원문 변경으로 stale 상태가 된 조항을 Spring에서 최대 100개 조회한다. 조항 원문을 한 번의 Embeddings API 배치 요청으로 변환한 뒤 `contentHash`가 여전히 일치하는 조항만 Spring에 저장한다. 호출 자체를 예약하는 스케줄러와 운영 정책 데이터 입력은 아직 포함하지 않는다.
- 앱 로그에는 질문·후보 본문·인증값 대신 임의 요청 ID, 시간, 결과 상태를 기록한다. 프록시/APM의 별도 본문 로깅도 확인해야 한다.

## 테스트와 평가

Spring 테스트는 별도 백엔드 저장소 루트에서 실행한다:

```powershell
.\gradlew.bat test
```

PostGIS·pgvector 테스트는 `meetple-postgres:16-3.5-bigm-vector0.8.6` 이미지를 사용한다. Docker가 없으면 건너뛰며 기존 Spring 보안 테스트에는 Redis가 필요하다.

AI 테스트는 이 저장소 루트에서 실행한다:

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python evals/run.py
```

자동 테스트는 실제 LangGraph와 MCP HTTP 프로토콜, OpenAI SDK의 파싱을 사용하되 외부 HTTP 응답을 대체한다. 유료 API를 호출하지 않는다.

`evals/cases.json`은 모임 검색의 날짜 해석, 초보자 근거, 빈 결과, 반경과 미지원 요청을 평가한다. `evals/moderation_cases.json`은 대표 신고, 안전 위협, 근거 부족과 프롬프트 인젝션 사례에서 신고 유형·위험도·제재·증거·정책 선택을 평가한다. 기본 실행은 두 평가 묶음의 데이터 형식만 검사하므로 **모델 품질 통과 결과가 아니다.** 라이브 평가도 가상 데이터와 메모리 내 검색 도구를 사용하므로 Spring SQL과 실제 DB의 하이브리드 검색을 검증하지는 않는다.

키와 모델 설정 후 아래 명령은 모임 검색과 신고 분석 전체 문항에 유료 API를 호출한다. 먼저 평가 묶음과 문항 수를 제한해 확인할 수 있다.

```powershell
.venv\Scripts\python evals/run.py --live
.venv\Scripts\python evals/run.py --live --suite moderation --limit 1
```

특정 문항 하나만 확인하려면 `--case weekend_beginner` 또는 `--case moderation_prompt_injection`처럼 평가 문항 ID를 지정한다.

평가 결과는 Git에서 제외된 `evals/results/`에 모델 ID, 검색 또는 신고 분석 조건 일치 여부, 문항별 시간, p50/p95로 기록한다. 가상 데이터와 대체 검색 도구를 사용하므로 Spring/MCP/실제 DB 성능 측정과 구분한다. 원문 검증과 위험도별 제재 제한은 실제 그래프에서 실행한다. 작은 고정 문항 세트의 결과를 전체 서비스 정확도로 해석하지 않는다.
