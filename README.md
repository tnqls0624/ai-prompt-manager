# 🧠 FastMCP Prompt Enhancement Server

Cursor IDE를 위한 **FastMCP 기반 Python MCP 서버**입니다. SSE(Server-Sent Events) 방식으로 실시간 통신하며 프롬프트 최적화, 벡터 검색, 대화 학습 등의 기능을 제공합니다.

## 🚀 **주요 특징**

### 🔥 **FastMCP 기반**

- **공식 MCP 라이브러리**: 표준 MCP 2024-11-05 스펙 완전 준수
- **SSE 실시간 통신**: Server-Sent Events로 빠른 응답
- **자동 도구 등록**: 데코레이터 기반 간편한 도구 정의
- **내장 오류 처리**: 안정적인 예외 처리 및 복구

### 🧠 **AI 프롬프트 최적화**

- **컨텍스트 기반 개선**: 프로젝트별 맞춤형 프롬프트 생성
- **벡터 검색 기반**: 유사한 성공 패턴 자동 탐지
- **학습 기반 추천**: 과거 대화 데이터 기반 개선 제안
- **실시간 피드백**: 사용자 선호도 즉시 반영

### 📊 **벡터 데이터베이스**

- **ChromaDB 통합**: 고성능 벡터 저장 및 검색
- **의미적 검색**: 임베딩 기반 유사도 계산
- **프로젝트 격리**: 프로젝트별 독립적인 컨텍스트 관리
- **자동 임베딩**: OpenAI 모델을 통한 자동 벡터화

### 🔧 **8개 핵심 도구**

1. **enhance_prompt**: 프롬프트 분석 및 개선
2. **store_prompt**: 프롬프트 벡터 저장
3. **store_conversation**: 대화 학습 데이터 저장
4. **search_similar_prompts**: 유사 프롬프트 검색
5. **search_similar_conversations**: 유사 대화 검색
6. **analyze_conversation_patterns**: 패턴 분석
7. **get_server_status**: 서버 상태 모니터링
8. **create_enhanced_prompt**: 프롬프트 템플릿 생성

## 🏗️ 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Cursor    │◄──►│  FastMCP    │◄──►│ LangChain +     │
│    IDE      │    │   Server    │    │ ChromaDB        │
└─────────────┘    └─────────────┘    └─────────────────┘
     stdio           SSE (HTTP)         Vector Store +
   (JSON-RPC)       (MCP Protocol)     OpenAI Embeddings
```

## 📦 설치 및 실행

### 필수 요구사항

- **Python 3.11+**
- **OpenAI API Key**
- **FastMCP 라이브러리**

### 1. 간단한 설치 및 실행

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
export OPENAI_API_KEY=your_openai_api_key_here

# FastMCP 서버 실행
./start_fastmcp_server.sh
```

### 2. 수동 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# 환경 변수 설정
export OPENAI_API_KEY=your_openai_api_key_here

# 서버 직접 실행
python mcp_server.py
```

### 3. 테스트 실행

```bash
# 기능 테스트
python tests/test_fastmcp_server.py
```

## 🐳 Docker 실행

### Docker로 간편 실행

더 안정적이고 격리된 환경에서 실행하려면 Docker를 사용하세요:

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY=your_openai_api_key_here

# Docker 컨테이너 빌드 및 시작
./scripts/start_docker_mcp.sh

# 서버 중지
./scripts/stop_docker_mcp.sh
```

### Docker Compose 수동 실행

```bash
# 환경 변수 설정
export OPENAI_API_KEY=your_openai_api_key_here

# 빌드 및 시작
docker-compose up -d

# 로그 확인
docker logs fastmcp-prompt-enhancement -f

# 정지
docker-compose down
```

### Cursor에서 Docker 서버 사용

Docker 기반 서버를 Cursor에서 사용하려면:

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "fastmcp-prompt-enhancement",
        "python",
        "mcp_server.py"
      ]
    }
  }
}
```

또는 매번 새 컨테이너로 실행:

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-v",
        "$(pwd)/data/chroma:/chroma/chroma",
        "-v",
        "$(pwd)/data/logs:/app/logs",
        "-e",
        "OPENAI_API_KEY=$OPENAI_API_KEY",
        "mcp-server_fastmcp-server",
        "python",
        "mcp_server.py"
      ]
    }
  }
}
```

### 📊 서비스 모니터링

```bash
# 컨테이너 상태 확인
docker ps | grep "fastmcp\|chromadb"

# 헬스체크 상태
docker inspect fastmcp-prompt-enhancement | grep Health -A 10
docker inspect chromadb-server | grep Health -A 10

# 로그 실시간 모니터링
docker logs fastmcp-prompt-enhancement -f
docker logs chromadb-server -f
```

## 💾 데이터 영속성 및 백업

### 🔒 안전한 데이터 저장

기본 설정에서는 **호스트 경로 바인딩**을 사용하여 데이터 손실을 방지합니다:

```
./data/chroma/     # ChromaDB 벡터 데이터
./data/logs/       # 애플리케이션 로그
./backups/chroma/  # 백업 파일
```

### 🗂️ 데이터 구조

```
mcp-server/
├── data/
│   ├── chroma/        # 🔐 ChromaDB 데이터 (영구 보존)
│   └── logs/          # 📝 서버 로그 파일
├── backups/
│   └── chroma/        # 💾 백업 파일들
└── docker-compose.yml # 🐳 기본 설정 (호스트 바인딩)
```

### 📥 백업하기

```bash
# 자동 백업 (3가지 방법 지원)
./scripts/backup_chroma.sh

# 백업 파일 예시
backups/chroma/
├── chroma_backup_20241218_143000.tar.gz  # 호스트 경로 백업
├── chroma_backup_20241218_143100.tar.gz  # Docker Volume 백업
└── chroma_backup_20241218_143200.tar.gz  # API 덤프 백업
```

### 📤 복원하기

```bash
# 백업에서 복원
./scripts/restore_chroma.sh

# 복원 방법 선택:
# 1. 호스트 경로 복원 (기본 설정) ⭐
# 2. Docker Volume 복원 (레거시 설정)
# 3. 데이터베이스 복원 (API 방식)
```

### 🏭 프로덕션 환경

더 안전한 프로덕션 설정을 원한다면:

```bash
# 프로덕션용 Docker Compose 사용
docker-compose -f docker-compose.production.yml up -d

# 특징:
# - 헬스체크 포함
# - 호스트 경로 바인딩
# - 백업 볼륨 자동 마운트
# - 강화된 모니터링
```

### 🚨 데이터 안전성

- **컨테이너 재시작**: 데이터 유지 ✅
- **Docker Compose 재시작**: 데이터 유지 ✅
- **시스템 재부팅**: 데이터 유지 ✅
- **컨테이너 삭제**: 데이터 유지 ✅
- **볼륨 삭제**: 데이터 유지 ✅ (호스트 바인딩)

## 🔧 Cursor 설정

### MCP 서버 연결

Cursor의 설정에 다음을 추가하세요:

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/your/mcp-server"
    }
  }
}
```

### 환경 변수 설정

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

## 🛠️ 사용 가능한 도구들 (16개)

### 🔥 **프롬프트 개선 도구**

#### 1. **enhance_prompt** 📝

```
프롬프트를 분석하고 개선 제안을 제공합니다.

입력:
- prompt: 개선할 프롬프트
- project_id: 프로젝트 ID (기본값: "default")
- context_limit: 컨텍스트 제한 (기본값: 5)

출력:
- enhanced_prompt: 개선된 프롬프트
- suggestions: 개선 제안사항
- context_used: 사용된 컨텍스트
```

#### 2. **get_prompt_recommendations** 🎯

```
프롬프트에 대한 추천사항을 조회합니다.

입력:
- prompt: 분석할 프롬프트
- project_id: 프로젝트 ID

출력:
- recommendations: 프롬프트 추천사항
```

### 📊 **대화 및 학습 도구**

#### 3. **store_conversation** 💾

```
사용자와 AI의 대화를 학습 데이터로 저장합니다.

입력:
- user_prompt: 사용자 프롬프트
- ai_response: AI 응답
- project_id: 프로젝트 ID

출력:
- success: 저장 성공 여부
- message: 결과 메시지
```

#### 4. **search_similar_conversations** 🔍

```
유사한 대화나 프롬프트를 검색합니다.

입력:
- query: 검색할 쿼리
- project_id: 프로젝트 ID
- limit: 결과 개수 제한

출력:
- results: 검색 결과 목록
- total_results: 총 결과 개수
```

#### 5. **analyze_conversation_patterns** 📈

```
대화 패턴을 분석하고 인사이트를 제공합니다.

입력:
- project_id: 프로젝트 ID

출력:
- 패턴 분석 결과
```

### 🔍 **프로젝트 인덱싱 도구**

#### 6. **network_project_upload** 🚀

```
네트워크 기반 고성능 프로젝트 업로드 및 인덱싱

입력:
- project_path: 업로드할 프로젝트 경로
- project_id: 프로젝트 ID
- max_workers: 병렬 워커 수 (기본값: 20)
- batch_size: 배치 크기 (기본값: 200)

출력:
- 업로드 결과 및 성능 통계
```

### 📁 **프로젝트 파일 관리**

#### 7. **search_project_files** 🔎

```
프로젝트 파일 내용에서 검색합니다.

입력:
- query: 검색할 내용
- project_id: 프로젝트 ID
- file_type: 파일 타입 필터 ("code", "documentation", "all")
- limit: 결과 개수 제한

출력:
- 파일 검색 결과
```

#### 8. **get_project_context_info** 📋

```
프로젝트 컨텍스트 정보를 조회합니다.

입력:
- project_id: 프로젝트 ID

출력:
- 프로젝트 컨텍스트 정보
```

### 📈 **고급 분석 도구**

#### 9. **analyze_prompt_patterns** 🧠

```
프로젝트의 프롬프트 패턴을 클러스터링으로 분석합니다.

입력:
- project_id: 프로젝트 ID
- n_clusters: 클러스터 개수 (기본값: 5)

출력:
- 클러스터링 분석 결과
```

#### 10. **extract_prompt_keywords** 🔤

```
프로젝트 프롬프트에서 중요한 키워드를 TF-IDF로 추출합니다.

입력:
- project_id: 프로젝트 ID
- max_features: 추출할 최대 키워드 수 (기본값: 20)

출력:
- 키워드 추출 결과
```

#### 11. **analyze_prompt_trends** 📊

```
프로젝트의 프롬프트 트렌드를 분석합니다.

입력:
- project_id: 프로젝트 ID

출력:
- 트렌드 분석 결과
```

### 🔄 **피드백 시스템**

#### 12. **submit_user_feedback** 👍

```
사용자 피드백 제출

입력:
- enhancement_id: 개선된 프롬프트 ID
- original_prompt: 원본 프롬프트
- enhanced_prompt: 개선된 프롬프트
- feedback_type: 피드백 타입 (accept, reject, partial_accept, modify)
- user_rating: 사용자 평점 (1-5)
- execution_success: 실행 성공 여부

출력:
- 피드백 분석 결과
```

#### 13. **get_feedback_statistics** 📊

```
프로젝트별 피드백 통계 조회

입력:
- project_id: 프로젝트 ID

출력:
- 피드백 통계 정보
```

#### 14. **analyze_feedback_patterns** 🔍

```
프로젝트별 피드백 패턴 분석

입력:
- project_id: 프로젝트 ID

출력:
- 피드백 패턴 분석 결과
```

### 🔧 **성능 및 모니터링**

#### 15. **get_fast_indexing_stats** ⚡

```
고속 인덱싱 서비스의 성능 통계를 반환합니다.

출력:
- 성능 설정 및 통계 정보
- 최적화 기능 목록
```

#### 16. **get_server_status** 🖥️

```
서버 상태 정보를 반환합니다.

출력:
- 서버 상태 정보
- 서비스 상태
- 지원 기능 목록
```

## 🧪 테스트 및 디버깅

### 🔍 **코드 인덱싱 상태 확인**

#### 1. Docker 컨테이너 상태 확인

```bash
# 모든 서비스 상태 확인
docker ps | grep -E "(fastmcp|chromadb|deepseek)"

# 결과 예시:
# fastmcp-prompt-enhancement    Running
# chromadb-server               Running
# deepseek-r1-server           Running (unhealthy)
```

#### 2. MCP 서버 헬스체크

```bash
# 서버 상태 확인
curl -s http://localhost:8000/api/v1/heartbeat | jq .

# 결과 예시:
{
  "status": "healthy",
  "message": "MCP 서버가 정상 작동 중입니다",
  "services": {
    "vector_service": true,
    "enhancement_service": true,
    "file_indexing_service": true,
    "fast_indexing_service": true,
    "analytics_service": true
  }
}
```

#### 3. 자동 인덱싱 상태 확인 (MCP 도구 사용)

```bash
# Cursor에서 MCP 도구 호출
get_auto_indexing_status

# 결과 예시:
{
  "success": true,
  "status": {
    "is_running": true,
    "last_scan_time": "2024-12-19T10:30:00Z",
    "projects_found": 2,
    "files_indexed": 1247,
    "indexing_progress": 100
  }
}
```

#### 4. 프로젝트 컨텍스트 확인

```bash
# 인덱싱된 프로젝트 정보 확인
get_project_context_info

# 특정 프로젝트 확인
get_project_context_info(project_id="lovechedule-app")
```

#### 5. 로그 확인

```bash
# 실시간 로그 모니터링
docker logs fastmcp-prompt-enhancement -f

# 최근 로그 확인
docker logs fastmcp-prompt-enhancement --tail 50
```

### 🚀 **인덱싱 문제 해결**

#### 프로젝트가 인덱싱되지 않는 경우

```bash
# 1. 프로젝트 마운트 확인
docker exec fastmcp-prompt-enhancement ls -la /host_projects/

# 2. 인덱싱 로그 확인
docker logs fastmcp-prompt-enhancement | grep -i "인덱싱\|indexing"

# 3. 컨테이너 재시작 (자동 재인덱싱)
docker-compose restart fastmcp-server
```

#### 인덱싱 성능 문제

```bash
# 1. 인덱싱 성능 통계 확인
get_fast_indexing_stats

# 2. 네트워크 업로드 사용 (대량 파일)
network_project_upload("/host_projects/large-project")
```

### 기능 테스트

```bash
# 전체 기능 테스트
python test_fastmcp_server.py

# 개별 도구 테스트
python -c "
import asyncio
from mcp_server import enhance_prompt, initialize_services

async def test():
    await initialize_services()
    result = await enhance_prompt('React 컴포넌트 만들기')
    print(result)

asyncio.run(test())
"
```

### 디버그 모드 실행

```bash
# 디버그 로그 활성화
LOG_LEVEL=DEBUG python mcp_server.py
```

### 📊 **인덱싱 통계 확인**

#### 프로젝트별 인덱싱 현황

```bash
# MCP 도구를 통한 통계 확인
get_server_status  # 전체 서비스 상태
get_project_context_info  # 프로젝트 컨텍스트 정보
analyze_prompt_patterns  # 프롬프트 패턴 분석
```

#### 벡터 DB 상태 확인

```bash
# ChromaDB 상태 확인
curl -s http://localhost:8001/api/v1/heartbeat

# 벡터 DB 컬렉션 확인
docker exec chromadb-server chroma list collections
```

## 📁 파일 구조

```
mcp-server/
├── mcp_server.py                     # 🔥 FastMCP 서버 메인 파일
├── config.py                         # ⚙️ 설정 파일
├── requirements.txt                  # 📦 Python 의존성
│
├── services/                         # 🛠️ 핵심 서비스들
│   ├── vector_service.py             # 📊 벡터 DB 서비스 (ChromaDB)
│   ├── prompt_enhancement_service.py # 🧠 프롬프트 개선 서비스
│   ├── advanced_analytics.py         # 📈 고급 분석 서비스 (scikit-learn)
│   └── file_indexing_service.py      # 📂 파일 인덱싱 서비스
│
├── models/                           # 📋 데이터 모델
│   ├── prompt_models.py              # 기본 프롬프트 모델
│   └── enhanced_models.py            # 고급 분석 모델
│
├── data/                             # 🗄️ 영구 데이터 저장소
│   ├── chroma/                       # ChromaDB 벡터 데이터
│   └── logs/                         # 애플리케이션 로그
│
├── backups/                          # 💾 백업 디렉토리
│   └── chroma/                       # ChromaDB 백업 파일들
│
├── scripts/                          # 🔧 관리 스크립트들
│   ├── start_docker_mcp.sh           # Docker 서버 시작
│   ├── stop_docker_mcp.sh            # Docker 서버 중지
│   ├── backup_chroma.sh              # 데이터 백업
│   └── restore_chroma.sh             # 데이터 복원
│
├── docker-compose.yml                # 🐳 기본 Docker 설정
├── docker-compose.production.yml     # 🏭 프로덕션 Docker 설정
├── Dockerfile                        # 🐳 컨테이너 빌드 설정
│
└── tests/                            # 🧪 테스트 파일들
    ├── test_fastmcp_server.py        # FastMCP 서버 테스트
    ├── test_mcp_system.py            # MCP 시스템 통합 테스트
    └── test_advanced_analytics.py    # 고급 분석 기능 테스트
```

## 🔍 MCP 도구 사용 흐름

### 1. 🚀 **초기 설정 및 프로젝트 확인**

#### 1단계: 서버 상태 확인

```
도구: get_server_status
목적: 서버 상태 및 서비스 초기화 확인
```

#### 2단계: 자동 인덱싱 결과 확인

```
도구: get_project_context_info
목적: 프로젝트 컨텍스트 정보 확인 (자동 인덱싱 완료 확인)
```

**💡 참고**: Docker 시작 시 자동으로 모든 프로젝트가 인덱싱됩니다!

### 2. 🧠 **프롬프트 개선 워크플로우**

#### 기본 프롬프트 개선

```
1. enhance_prompt 호출
   입력: "React 컴포넌트 만들기"

2. 결과 분석
   출력: "TypeScript를 사용하여 재사용 가능한 React 버튼 컴포넌트를 작성하세요.
         props로 size, variant, disabled 상태를 받고,
         접근성을 고려한 ARIA 속성을 포함해주세요."
```

#### 유사한 프롬프트 검색

```
3. search_similar_conversations 호출
   입력: "React 컴포넌트"
   목적: 과거 성공적인 패턴 참조
```

#### 프롬프트 추천 확인

```
4. get_prompt_recommendations 호출
   입력: "React 컴포넌트 만들기"
   목적: 추가 개선 제안 받기
```

### 3. 📊 **대화 학습 및 피드백**

#### 성공적인 대화 저장

```
5. store_conversation 호출
   입력:
   - user_prompt: "React 컴포넌트 만들기"
   - ai_response: "생성된 코드"
   목적: 학습 데이터 축적
```

#### 피드백 제출

```
6. submit_user_feedback 호출
   입력:
   - feedback_type: "accept"
   - user_rating: 5
   - execution_success: true
   목적: 개선 품질 학습
```

### 4. 🔍 **프로젝트 분석 및 검색**

#### 프로젝트 파일 검색

```
7. search_project_files 호출
   입력:
   - query: "authentication"
   - file_type: "code"
   목적: 관련 코드 찾기
```

#### 패턴 분석

```
8. analyze_prompt_patterns 호출
   목적: 프로젝트 내 프롬프트 패턴 분석
```

#### 키워드 추출

```
9. extract_prompt_keywords 호출
   목적: 자주 사용되는 키워드 확인
```

### 5. 🔄 **지속적인 개선**

#### 피드백 통계 확인

```
10. get_feedback_statistics 호출
    목적: 프로젝트별 피드백 현황 파악
```

#### 트렌드 분석

```
11. analyze_prompt_trends 호출
    목적: 프롬프트 사용 트렌드 분석
```

#### 성능 최적화

```
12. get_fast_indexing_stats 호출
    목적: 인덱싱 성능 확인
```

### 🎯 **실제 사용 시나리오**

#### 시나리오 1: 새 프로젝트 시작

```
1. get_server_status → 서버 상태 확인
2. get_project_context_info → 자동 인덱싱 결과 확인
3. enhance_prompt → 첫 번째 프롬프트 개선
4. store_conversation → 대화 저장
```

#### 시나리오 2: 기존 프로젝트 개선

```
1. search_project_files → 관련 파일 찾기
2. search_similar_conversations → 과거 대화 검색
3. enhance_prompt → 컨텍스트 기반 프롬프트 개선
4. submit_user_feedback → 결과 피드백
```

#### 시나리오 3: 프로젝트 분석

```
1. analyze_prompt_patterns → 패턴 분석
2. extract_prompt_keywords → 키워드 추출
3. analyze_prompt_trends → 트렌드 분석
4. get_feedback_statistics → 피드백 통계
```

### 💡 **최적 사용 팁**

1. **프로젝트 ID 일관성**: 같은 프로젝트는 동일한 project_id 사용
2. **자동 인덱싱**: 새 프로젝트는 Docker 시작 시 자동으로 인덱싱됨
3. **피드백 제공**: 개선된 프롬프트 사용 후 피드백 제출
4. **정기 분석**: 주기적으로 패턴 분석 및 통계 확인
5. **자동 처리**: 대부분의 인덱싱은 자동으로 처리됨

### 🔧 **Docker 환경에서의 자동 인덱싱**

#### 자동 인덱싱 확인 방법

```bash
# 1. 컨테이너 상태 확인
docker ps | grep fastmcp

# 2. 자동 인덱싱 로그 확인
docker logs fastmcp-prompt-enhancement | grep -i "인덱싱\|indexing"

# 3. 프로젝트 마운트 확인
# docker-compose.yml의 volumes 섹션에서
# /host_projects/your-project:/host_projects/your-project:ro
```

#### 자동 인덱싱 동작 원리

```
✅ Docker 컨테이너 시작 → 자동 인덱싱 서비스 시작
✅ /host_projects/ 디렉토리 자동 탐지
✅ 모든 프로젝트 파일 자동 스캔 및 인덱싱
✅ 5분마다 증분 스캔 (변경된 파일만)
```

**💡 참고**: 모든 인덱싱이 자동으로 처리되므로 수동 개입이 불필요합니다!

### 📋 **프로젝트별 사용 예시**

#### React 프로젝트

```
1. get_project_context_info(project_id="react-app")
2. search_project_files("useState hook")
3. enhance_prompt("React 상태 관리 컴포넌트 만들기")
4. submit_user_feedback(feedback_type="accept")
```

#### Node.js API 프로젝트

```
1. get_project_context_info(project_id="api-server")
2. search_project_files("express middleware")
3. enhance_prompt("REST API 엔드포인트 만들기")
4. analyze_prompt_patterns()
```

#### 다중 프로젝트 관리

```
1. network_project_upload("project-a", project_id="project-a")
2. network_project_upload("project-b", project_id="project-b")
3. get_feedback_statistics("project-a")
4. analyze_feedback_patterns("project-b")
```

## 🚀 고급 기능

### 프롬프트 템플릿

```python
# 사용자 정의 프롬프트 템플릿 생성
template = create_enhanced_prompt(
    topic="React 컴포넌트 개발",
    context="TypeScript + Tailwind CSS 프로젝트"
)
```

### 리소스 접근

```
# 프로젝트 히스토리 조회
prompt-history://projects/my-project

# 서버 상태 정보
server-info://status
```

## 🔧 환경 변수

```bash
# 필수
OPENAI_API_KEY=your_openai_api_key_here

# 선택적
LOG_LEVEL=INFO                    # 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
CHROMA_DB_PATH=./chroma_db        # ChromaDB 데이터 경로
MCP_SERVER_NAME=Prompt Enhancement MCP Server  # 서버 이름
```

->

## 🔧 환경 변수

### 📝 .env 파일 생성 (권장)

프로젝트 루트에 `.env` 파일을 생성하여 환경변수를 관리하세요:

```bash
# .env 파일 예시
# 🔐 OpenAI API 설정 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# 📝 로깅 설정
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR

# 🧠 AI 설정
MAX_CONTEXT_LENGTH=5              # 컨텍스트 최대 개수
SIMILARITY_THRESHOLD=0.7          # 유사도 임계값 (0.0 ~ 1.0)

# 📈 분석 설정
ENABLE_ADVANCED_ANALYTICS=true    # 고급 분석 기능 활성화
CLUSTERING_ALGORITHM=kmeans       # 클러스터링 알고리즘
MAX_CLUSTERS=10                   # 최대 클러스터 개수

# 🚀 MCP 설정
MCP_SERVER_NAME=FastMCP Prompt Enhancement Server
MCP_VERSION=2.0.0
```

### 🌐 전체 환경변수 목록

| 변수명                      | 기본값                            | 설명                                 |
| --------------------------- | --------------------------------- | ------------------------------------ |
| `OPENAI_API_KEY`            | None                              | **필수** OpenAI API 키               |
| `LOG_LEVEL`                 | INFO                              | 로그 레벨 (DEBUG/INFO/WARNING/ERROR) |
| `MAX_CONTEXT_LENGTH`        | 5                                 | 프롬프트 향상시 참조할 컨텍스트 개수 |
| `SIMILARITY_THRESHOLD`      | 0.7                               | 유사도 검색 임계값                   |
| `ENABLE_ADVANCED_ANALYTICS` | true                              | scikit-learn 기반 고급 분석          |
| `CLUSTERING_ALGORITHM`      | kmeans                            | 프롬프트 클러스터링 알고리즘         |
| `MAX_CLUSTERS`              | 10                                | 클러스터링 최대 그룹 수              |
| `MCP_SERVER_NAME`           | FastMCP Prompt Enhancement Server | MCP 서버 이름                        |

### 🐳 Docker 환경에서 사용

```bash
# 환경변수 설정 후 Docker 실행
export OPENAI_API_KEY=your_key_here
export LOG_LEVEL=DEBUG
./scripts/start_docker_mcp.sh
```

## 📊 성능 및 한계

### 성능 지표

- **응답 속도**: 평균 200-500ms (OpenAI API 포함)
- **동시 연결**: 단일 Cursor 인스턴스 지원
- **메모리 사용량**: 약 100-200MB (벡터 DB 포함)
- **저장 용량**: 프롬프트당 약 1-2KB

### 알려진 한계

- 현재 OpenAI API만 지원 (다른 LLM 모델 지원 예정)
- 단일 사용자 환경에 최적화
- 패턴 분석 기능은 개발 중

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🆘 문제 해결

### 자주 발생하는 문제

1. **FastMCP 모듈을 찾을 수 없음**

   ```bash
   pip install fastmcp
   ```

2. **OpenAI API 키 오류**

   ```bash
   export OPENAI_API_KEY=your_actual_api_key
   ```

3. **ChromaDB 권한 오류**

   ```bash
   chmod -R 755 chroma_db/
   ```

4. **포트 충돌**
   - FastMCP는 자동으로 사용 가능한 포트를 선택합니다
   - 필요시 `config.py`에서 포트 설정 변경

### 로그 확인

```bash
# 상세 로그로 실행
LOG_LEVEL=DEBUG python mcp_server.py

# 로그 파일로 저장
python mcp_server.py 2>&1 | tee mcp_server.log
```

## 🎯 **MCP 도구 사용 시 주의사항**

### ⚠️ **중요 사항**

1. **프로젝트 ID 일관성**

   - 같은 프로젝트는 항상 동일한 `project_id` 사용
   - 예: `"lovechedule-app"`, `"api-backend"` 등

2. **자동 인덱싱 확인**

   - Docker 시작 시 자동으로 모든 프로젝트가 인덱싱됨
   - `get_project_context_info`로 인덱싱 결과 확인

3. **피드백 제공**

   - 개선된 프롬프트 사용 후 반드시 피드백 제출
   - 학습 품질 향상을 위해 정확한 평가 필요

4. **임베딩 모델 설정**
   - DeepSeek R1 사용 시: 로컬 처리, 무료, 느림
   - OpenAI API 사용 시: 외부 API, 유료, 빠름

### 🔧 **성능 최적화 팁**

1. **배치 처리**

   - 대량 파일: `network_project_upload` 사용
   - 일반 파일: 자동 인덱싱 사용 (Docker 시작 시)

2. **캐싱 활용**

   - 검색 결과는 5분간 캐싱됨
   - 동일한 쿼리는 빠른 응답

3. **병렬 처리**
   - 최대 20개 파일 동시 처리
   - 배치 크기: 200개 권장

### 📋 **문제 해결 체크리스트**

#### 인덱싱 문제

```
□ Docker 컨테이너 실행 상태 확인
□ 프로젝트 마운트 경로 확인
□ 자동 인덱싱 로그 확인
□ ChromaDB 연결 상태 확인
□ 컨테이너 재시작 시도
```

#### 프롬프트 개선 문제

```
□ 프로젝트 자동 인덱싱 완료 여부 확인
□ 컨텍스트 정보 존재 여부 확인
□ DeepSeek 임베딩 모델 정상 작동 확인
□ 피드백 데이터 충분성 확인
```

#### 성능 문제

```
□ 자동 인덱싱 로그 확인
□ 메모리 사용량 확인
□ DeepSeek 모델 연결 상태 확인
□ ChromaDB 용량 확인
```

### 🚀 **최적 워크플로우**

#### 일반적인 사용 패턴

```
1. 프로젝트 설정
   → get_server_status
   → get_project_context_info

2. 프롬프트 개선
   → search_similar_conversations
   → enhance_prompt
   → store_conversation

3. 피드백 및 학습
   → submit_user_feedback
   → analyze_feedback_patterns

4. 정기 분석
   → analyze_prompt_patterns
   → extract_prompt_keywords
   → analyze_prompt_trends
```

---

💡 **팁**: 현재 DeepSeek R1 모델을 사용하고 있으므로 OpenAI API 키 없이도 모든 기능을 사용할 수 있습니다!
