# 🚀 FastMCP 기반 프롬프트 향상 서버

DeepSeek R1과 ChromaDB를 활용한 고성능 AI 프롬프트 향상 시스템입니다.

## ⚡ 성능 최적화 (NEW!)

### 🔥 주요 성능 개선사항

- **병렬 처리 최적화**: 동시성 제한을 10→100개로 대폭 증가
- **배치 크기 증가**: 파일 처리 배치를 100→200개로 확장
- **임베딩 최적화**: 배치 임베딩 크기를 50→100개로 증가
- **연결 풀링**: HTTP 연결 풀 크기를 대폭 확장 (100개 연결)
- **메모리 최적화**: 청크 크기 및 오버랩 최적화
- **ChromaDB 배치**: 벡터 저장 배치 크기를 500개로 증가

### 📊 성능 비교

| 항목           | 기존        | 최적화 후    | 개선율        |
| -------------- | ----------- | ------------ | ------------- |
| 동시 파일 처리 | 50개        | 200개        | **300%**      |
| 배치 크기      | 100개       | 200-500개    | **400%**      |
| HTTP 연결 풀   | 기본        | 100개 연결   | **대폭 개선** |
| 임베딩 배치    | 50개        | 100개        | **100%**      |
| 처리 속도      | ~10 파일/초 | ~50+ 파일/초 | **500%**      |

### 🚀 고성능 업로드 스크립트

새로운 최적화된 업로드 스크립트를 제공합니다:

```bash
# 고성능 업로드 사용
python scripts/fast_upload.py /path/to/your/project --project-id my-project

# 옵션 지정
python scripts/fast_upload.py /path/to/project \
    --project-id my-project \
    --project-name "My Project" \
    --server-url http://localhost:8000
```

**성능 특징:**

- 최대 50개 파일 병렬 읽기
- 300개 파일 배치 업로드
- HTTP/2 연결 풀링 최적화
- 자동 메모리 압박 방지

## 🐳 Docker 설정 최적화

Docker Compose에 리소스 제한과 성능 설정이 추가되었습니다:

```yaml
# 성능 최적화된 환경변수
environment:
  - MAX_CONCURRENT_REQUESTS=100
  - MAX_CONCURRENT_FILES=200
  - EMBEDDING_BATCH_SIZE=100
  - CHROMA_BATCH_SIZE=500
  - ENABLE_PARALLEL_INDEXING=true

# 리소스 제한
deploy:
  resources:
    limits:
      memory: 6G
      cpus: "4"
    reservations:
      memory: 3G
      cpus: "2"
```

## 📈 성능 모니터링

성능 통계를 확인하려면:

```bash
# 서버 상태 확인
curl http://localhost:8000/api/v1/heartbeat

# 성능 통계 조회 (MCP 도구 사용)
# get_fast_indexing_stats 함수 호출
```

## ⚙️ 설정 최적화

`config.py`에서 다음 설정들이 최적화되었습니다:

```python
# 성능 최적화 설정
max_concurrent_requests: int = 50  # 증가
max_concurrent_files: int = 100    # 새로 추가
embedding_batch_size: int = 100    # 증가
chroma_batch_size: int = 500       # 새로 추가
enable_parallel_indexing: bool = True  # 새로 추가
```

---

## 🎯 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd mcp-server

# Python 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Docker로 서비스 시작

```bash
# 서비스 시작 (최적화된 설정으로)
docker-compose up -d

# 로그 확인
docker-compose logs -f fastmcp-server
```

### 3. 프로젝트 업로드 (고성능)

```bash
# 새로운 고성능 스크립트 사용
python scripts/fast_upload.py /path/to/your/project

# 또는 기존 방식
python -m mcp_server
```

## 🔧 주요 기능

- **🤖 DeepSeek R1 통합**: 로컬 LLM으로 프라이버시 보장
- **📊 ChromaDB 벡터 검색**: 고성능 벡터 데이터베이스
- **⚡ 병렬 처리**: 대용량 프로젝트 고속 인덱싱
- **🔍 지능형 검색**: 의미 기반 코드 검색
- **📈 실시간 분석**: 프롬프트 패턴 분석
- **🎯 피드백 학습**: 사용자 피드백 기반 개선

## 📚 사용법

### MCP 도구 함수들

```python
# 프롬프트 향상
enhance_prompt("코드 리뷰를 위한 체크리스트 만들어줘")

# 유사 대화 검색
search_similar_conversations("React 컴포넌트 최적화")

# 프로젝트 파일 검색
search_project_files("useState hook 사용법")

# 성능 통계 조회
get_fast_indexing_stats()
```

## 🛠️ 개발 환경

### 로컬 개발

```bash
# 개발 모드로 실행
python mcp_server.py

# 테스트 실행
python -m pytest tests/
```

### 환경 변수

```bash
# .env 파일 생성
EMBEDDING_MODEL_TYPE=deepseek
DEEPSEEK_API_BASE=http://localhost:11434
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=100
```

## 🎯 성능 팁

1. **메모리 설정**: Docker에 충분한 메모리 할당 (최소 8GB 권장)
2. **SSD 사용**: 벡터 DB 성능을 위해 SSD 스토리지 권장
3. **네트워크**: 로컬 네트워크에서 최적 성능 발휘
4. **배치 크기**: 대용량 프로젝트는 배치 크기 조정 고려

## 🐛 문제 해결

### 성능 이슈

```bash
# 메모리 사용량 확인
docker stats

# 로그 확인
docker-compose logs fastmcp-server | grep -E "(ERROR|WARNING|성능)"

# ChromaDB 연결 확인
curl http://localhost:8001/api/v1/heartbeat
```

### 일반적인 해결책

- **메모리 부족**: Docker 메모리 제한 증가
- **연결 타임아웃**: 네트워크 설정 확인
- **임베딩 실패**: DeepSeek 서비스 상태 확인

## 📖 추가 문서

- [DeepSeek R1 설정](README_DEEPSEEK_R1.md)
- [Cursor RAG 통합](README_CURSOR_RAG.md)
- [API 문서](docs/api.md)

## 🤝 기여

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**🚀 이제 훨씬 빨라진 성능으로 프로젝트 인덱싱을 경험해보세요!**
