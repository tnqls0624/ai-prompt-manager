# FastMCP 프롬프트 향상 서버

FastMCP, ChromaDB 벡터 저장소, Ollama 로컬 LLM을 활용한 프롬프트 매니징 서버입니다.
유저의 프롬프팅 및 히스토리를 저장하고 해당 데이터를 기반으로 유저의 요구사항에 대한 상세한 프롬프팅을 만들어서 LLM에 요청합니다.

## 아키텍처

```
Cursor/IDE → FastMCP Server (MCP + SSE)
                 ├─ MCP Tools
                 ├─ LangChain RAG Pipeline
                 │     ├─ Retriever → VectorService → ChromaDB
                 │     └─ LLM → Ollama
                 └─ Indexing/Watcher/Feedback Services
```

## LLM 사용

본 서버는 두 가지 모드로 동작합니다.

### 1) LLM 사용(전체 기능)

- 모델: `deepseek-r1:latest` (Ollama)
- 임베딩: `nomic-embed-text` (Nomic AI)
- 제공 기능:
  - AI 기반 프롬프트 향상
  - 컨텍스트 기반 코드 생성
  - 지능형 요약

### 2) LLM 미사용(폴백 모드)

- 템플릿 기반 향상으로 동작 지속
- `StandardPromptFormatter`를 사용한 구조화 개선
- 벡터 검색/컨텍스트 조회 기능 유지

## 코어 컴포넌트

### 1) MCP 도구(15개)

**LLM 활용 도구:**

- `enhance_prompt` - LLM 사용 시 AI 기반 프롬프트 향상
- `get_prompt_recommendations` - 컨텍스트 기반 추천
- `generate_test_skeleton` - 최소 실패 테스트 스켈레톤(LLM 폴백 지원)

**벡터 검색 도구(LLM 불필요):**

- `store_conversation` - 사용자-AI 상호작용 저장
- `search_similar_conversations` - 임베딩 기반 의미 검색
- `search_project_files` - 인덱싱된 프로젝트 파일 검색
- `get_project_context_info` - 프로젝트 컨텍스트 조회

**분석 도구(LLM 불필요):**

- `analyze_conversation_patterns` - 패턴 분석
- `analyze_prompt_patterns` - K-means 클러스터링
- `extract_prompt_keywords` - TF-IDF 키워드 추출
- `analyze_prompt_trends` - 시계열 트렌드 분석

**피드백 도구:**

- `submit_user_feedback` - 피드백 루프
- `get_feedback_statistics` - 지표/분석
- `analyze_feedback_patterns` - 패턴 인식

**시스템 도구:**

- `get_fast_indexing_stats` - 인덱싱 성능 지표
- `get_server_status` - 서버 헬스 체크

### 2) REST API 엔드포인트

**LLM 필수:**

- `/api/v1/rag/enhance-prompt` - LLM 기반 LangChain RAG 향상
- `/api/v1/rag/generate-code` - 코드 생성(주요 LLM 사용처)
- `/api/v1/rag/search-summarize` - 검색 + 요약

**LLM 선택:**

- `/api/v1/enhance-prompt-stream/{connection_id}` - 스트리밍 향상(폴백 지원)
- `/api/v1/sse/{connection_id}` - SSE 연결
- `/api/v1/upload-batch` - 배치 파일 업로드
- `/api/v1/watcher/start` - 파일 시스템 감시 시작
- `/api/v1/feedback` - 피드백 제출
- `/api/v1/heartbeat` - 서버 상태 체크
- `/api/v1/validate` - 시스템/인덱싱/LLM 종합 점검
- `/api/v1/resource/snippet` - 파일 경로/라인 범위로 스니펫 반환
- `/api/v1/rag/generate-edit` - 최소 JSON 에디트 생성
- `/api/v1/index/warmup/{project_id}` - 프로젝트별 TF-IDF/BM25 캐시 빌드
- `/api/v1/audit/recent` - 최근 감사 로그(JSONL)
- `/api/v1/audit/search` - 감사 로그 검색(프로젝트/이벤트/기간)
- `/metrics` - Prometheus 메트릭(선택)
- `/dashboard` - 간단 대시보드(자동 새로고침)

### 3) 서비스

- `VectorService` - ChromaDB + Nomic 임베딩 연동
- `PromptEnhancementService` - 프롬프트 향상(LLM 폴백)
- `FastIndexingService` - 병렬 파일 인덱싱
- `LangChainRAGService` - RAG 파이프라인 + LLM 연동
- `FileWatcherService` - 실시간 파일 감시
- `FeedbackService` - 사용자 피드백 처리
- `AdvancedAnalyticsService` - ML 분석

## 성능

실측 기준 고성능 최적화:

- 동시 처리: 100 요청
- 파일 처리: 200 파일 병렬
- 임베딩 배치: 100 문서/배치
- ChromaDB 배치: 500 벡터/쓰기
- 연결 풀: 100 지속 HTTP 커넥션
- LLM 요청: 비동기 + 타임아웃 처리
- 하이브리드 검색: 의미 + TF-IDF 병렬, 프로젝트 범위 캐시
- 재랭킹: 의미/키워드/최신성/복잡도 가중치 튜닝
- TF-IDF 인덱스: 프로젝트별 TTL 캐시
- 지속 캐시: `cache_dir` 아래 벡터라이저/행렬 저장

## 빠른 시작

### 1) Docker 사용(권장)

```bash
# 모든 서비스 시작(Ollama 포함)
docker-compose up -d

# Ollama 모델 로드 확인
docker exec deepseek-r1-server ollama list

# 미로딩 시 모델 풀(Pull)
docker exec deepseek-r1-server ollama pull deepseek-r1

# 로그 보기
docker-compose logs -f fastmcp-server
```

### 2) Python 직접 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 로컬 Ollama 실행 보장
ollama serve

# 모델 풀
ollama pull deepseek-r1
ollama pull nomic-embed-text

# 서버 실행
python mcp_server.py
```

## 설정

`docker-compose.yml`의 주요 환경변수:

```yaml
environment:
  # 모델 설정
  - EMBEDDING_MODEL_TYPE=deepseek # Ollama 백엔드 사용 구성명
  - DEEPSEEK_API_BASE=http://deepseek-r1:11434 # Ollama 엔드포인트
  - DEEPSEEK_EMBEDDING_MODEL=nomic-embed-text # Nomic 임베딩 모델
  - DEEPSEEK_LLM_MODEL=deepseek-r1:latest # LLM 모델

  # 성능 설정
  - MAX_CONCURRENT_REQUESTS=100
  - EMBEDDING_BATCH_SIZE=100
  - CHROMA_BATCH_SIZE=500

  # 하이브리드 검색 가중치(선택)
  - HYBRID_SEMANTIC_WEIGHT=0.7
  - HYBRID_KEYWORD_WEIGHT=0.3
  - RECENCY_WEIGHT=0.1
  - COMPLEXITY_WEIGHT=0.1
  - TFIDF_INDEX_TTL_SECONDS=300
  - CACHE_DIR=/data/cache

  # 시작 시 워밍업(선택)
  - WARMUP_ON_START=false
  - WARMUP_PROJECT_IDS=my-project,another-project

  # 감사 로그(선택)
  - AUDIT_LOG_ENABLED=false

  # 인증(기본 비활성)
  - REQUIRE_API_KEY=false
  - API_KEY=
  - JWT_ENABLED=false
  - JWT_SECRET=
  - JWT_ALGORITHMS=HS256
  - PROJECT_QUOTA_PER_MINUTE=0
```

참고: 환경변수 접두사로 "deepseek"를 사용하지만, 실제 모델은 아래와 같습니다.

- 임베딩: `nomic-embed-text` (약 1.5GB, 768차원)
- LLM: `deepseek-r1:latest` (Ollama)

## 프로젝트 업로드

### 고속 배치 업로드

```bash
python scripts/fast_upload.py /path/to/project --project-id my-project
```

특징:

- 파일 병렬 읽기(동시 50)
- 배치 API 호출(요청당 300 파일)
- 지수 백오프 자동 재시도
- 진행률 추적

## MCP 통합

### Cursor 설정

MCP 설정에 다음을 추가하세요:

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "EMBEDDING_MODEL_TYPE": "deepseek",
        "DEEPSEEK_EMBEDDING_MODEL": "nomic-embed-text",
        "DEEPSEEK_LLM_MODEL": "deepseek-r1:latest",
        "DEEPSEEK_API_BASE": "http://localhost:11434"
      }
    }
  }
}
```

### 사용 예시

#### LLM 사용

```python
# AI 기반 프롬프트 향상
result = await enhance_prompt(
    prompt="Build a React component",
    project_id="my-project",
    context_limit=5
)
# 반환: 컨텍스트가 반영된 개선 프롬프트
```

#### LLM 미사용(폴백)

```python
# 동일 호출이지만 템플릿 기반 향상 반환
result = await enhance_prompt(
    prompt="Build a React component",
    project_id="my-project",
    context_limit=5
)
# 반환: 컨텍스트 포함 구조화 템플릿(비 LLM)
```

#### 테스트 스켈레톤 생성(TDD Red)

#### 스트리밍 코드 생성(SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/rag/generate-code \
  -H 'content-type: application/json' \
  -d '{
        "prompt":"Implement user login API",
        "project_id":"my-project",
        "context_limit":5,
        "stream":true
      }'
```

#### 파일 스니펫 가져오기

```bash
curl "http://localhost:8000/api/v1/resource/snippet?file_path=/host_projects/myproj/app.py&start_line=10&end_line=60"
```

#### 최소 JSON 에디트 생성

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate-edit \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Rename function foo to fetch_user",
        "project_id":"my-project",
        "file_path":"/host_projects/myproj/app.py",
        "diff_context":"def foo(...):\n    pass\n"
      }'
```

```python
result = await generate_test_skeleton(
    feature="user can reset password via token",
    framework="pytest",  # 또는 "jest", "unittest"
    project_id="my-project"
)
print(result["content"])  # 생성된 테스트 파일 내용
```

## 기술 스택

- FastMCP 2.9.0 - MCP 프로토콜 구현
- ChromaDB 0.4.22 - 벡터 DB
- LangChain 0.1.5 - RAG/LLM 오케스트레이션
- Ollama - 로컬 LLM 서버
  - `deepseek-r1:latest` - 언어 모델
  - `nomic-embed-text` - 임베딩 모델
- scikit-learn 1.3.0 - ML(클러스터링/TF-IDF)
- SSE/WebSocket - 실시간 통신

## 리소스 요구사항

### 최소(LLM 미사용)

- 메모리: 2GB
- CPU: 2코어
- 스토리지: 1GB + 데이터

### 권장(LLM 사용)

- 메모리: 8GB 이상(모델 크기 의존)
- CPU: 4코어
- 스토리지: SSD 20GB+ (모델)
- Docker: 6GB 메모리 할당

## 개발

### 테스트 실행

```bash
python -m pytest tests/ -v
```

### 신규 MCP 도구 추가

```python
@mcp.tool()
async def your_new_tool(param: str) -> Dict[str, Any]:
    """Tool description"""
    # self.llm 사용 가능 시 활용
    if self.llm:
        result = await self.llm.arun(prompt)
    else:
        result = fallback_logic()
    return result
```

## 아키텍처 결정(ADR)

1. FastMCP 채택: 성능 우수, SSE 기본 지원
2. ChromaDB 채택: 로컬 벡터 DB 중 성능 우수
3. Ollama 사용: 로컬 실행/프라이버시/비용 절감
4. Nomic 임베딩: 오픈소스/효율/품질
5. 폴백 메커니즘: LLM 없이도 서비스 지속
6. 병렬 처리: 5~10배 성능 향상

## 모니터링

```bash
# LLM 가용성 확인
curl http://localhost:11434/api/tags

# 성능 상태
curl http://localhost:8000/api/v1/heartbeat

# 에러 추적
docker-compose logs fastmcp-server | grep ERROR

# ChromaDB 헬스
curl http://localhost:8001/api/v1/heartbeat

# 시스템 종합 점검(LLM/인덱싱/에러/성능)
curl "http://localhost:8000/api/v1/validate?project_id=my-project"

# Prometheus 메트릭(선택)
curl http://localhost:8000/metrics

# 대시보드(그라파나)
open http://localhost:3000
```

## 알려진 제한 사항

- 단일 파일 최대 50MB
- ChromaDB 컬렉션 최대 100만 벡터
- 동시 연결 100(설정 가능)
- LLM 컨텍스트 창: 모델 의존(통상 8K~32K 토큰)
- LLM 응답 시간: 프롬프트 복잡도에 따라 1~10초

## 트러블슈팅

### LLM 동작 안할시

```bash
# Ollama 상태 확인
curl http://localhost:11434/api/tags

# 모델 로드 확인
ollama list

# 모델 미설치 시 Pull
ollama pull deepseek-r1
```
