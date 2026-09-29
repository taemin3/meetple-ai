# Meetple AI

- 이 폴더는 `https://github.com/taemin3/meetple-ai`의 독립 Python 저장소다.
- FastAPI, LangGraph, MCP, OpenAI 코드와 AI 평가를 관리한다. Spring 데이터 조회·로그인 코드는 별도 `meetple-backend` 저장소 소관이다.
- Python 3.12 이상을 사용하며 명령은 이 저장소 루트에서 실행한다.
- 환경변수는 `.env.example`을 참고한다. 실제 `.env`, API 키, 서비스 키, 평가 결과는 커밋하지 않는다.
- 인증값·질문·후보 원문을 로그에 남기지 않는다. 사용자 JWT를 모델이나 AI 서버로 전달하지 않는다.
- 원본 후보 ID·원문 근거 검증과 검색 반경 제한을 보존한다.
- 검증: `python -m ruff check .`, `python -m ruff format --check .`, `python -m pytest -q`, `python evals/run.py`.
- `evals/run.py --live`는 유료 OpenAI 호출이다. 외부 호출 없는 테스트와 실제 모델 품질 평가를 구분해 보고한다.
- 초기 저장소 구성 이후 작업은 기능 브랜치에서 구현·검증·커밋·푸시한다. PR 생성/병합은 명시적으로 요청받았을 때만 수행한다.
- 기존 변경을 보존하고 작업과 관련된 파일만 커밋한다.
