# 🚀 Cursor RAG 지능형 프롬프트 시스템

Cursor 에디터와 LangChain RAG + Chroma 벡터 DB를 활용한 지능형 프롬프트 시스템입니다.

## 📋 주요 기능

### 1. 🔄 **실시간 파일 감지 및 자동 업로드**

- 사용자가 새로운 코드를 생성하거나 수정하면 자동으로 MCP 서버로 전송
- **로컬 파일 와처**: 호스트에서 실행되어 파일 변경 감지 (watchdog 기반)
- **네트워크 업로드**: Docker 컨테이너의 MCP 서버로 HTTP API 전송
- 중복 업로드 방지 (해시 기반 변경 감지)
- 지능형 파일 필터링 (node_modules, .git 등 제외)

### 2. 🧠 **LangChain RAG 파이프라인**

- DocumentLoader, TextSplitter, Retriever, PromptTemplate 활용
- 프로젝트 컨텍스트 기반 프롬프트 향상
- 의미 기반 검색 (Semantic Search)
- 컨텍스트 재사용 극대화

### 3. 🗄️ **Chroma 벡터 데이터베이스**

- 프로젝트 문서 및 코드 임베딩 저장
- 빠른 유사도 검색
- 프로젝트별 컨텍스트 관리

### 4. 🎯 **지능형 프롬프트 재작성**

- 사용자 프롬프트 + 관련 컨텍스트 결합
- 표준화된 프롬프트 포맷
- 프로젝트 일관성 유지

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
export OPENAI_API_KEY="your-openai-api-key"
export CHROMA_PERSIST_DIR="/data/chroma"
export LOG_DIR="/data/logs"
```

### 3. Docker 환경 실행

```bash
docker-compose up -d
```

## 📖 사용법

### 1. 초기 설정

```bash
# Cursor RAG 설정 파일 생성
python cursor_rag_client.py init

# 서버 상태 확인
python cursor_rag_client.py health
```

### 2. 프로젝트 업로드 (최초 1회)

```bash
# 현재 프로젝트 업로드
python cursor_rag_client.py upload --project-path ./your-project --project-id my-project

# 또는 기존 업로드 스크립트 사용
python upload_project.py --project-path ./your-project --project-id my-project
```

### 3. 파일 감시 시작 (자동 업로드)

```bash
# 파일 감시 시작 (일회성)
python cursor_rag_client.py watch --project-path ./your-project --project-id my-project

# 파일 감시 시작 (계속 실행, 백그라운드 모드)
python cursor_rag_client.py watch --project-path ./your-project --project-id my-project --keep-alive

# 감시 상태 확인
python cursor_rag_client.py status

# 감시 중지
python cursor_rag_client.py unwatch --project-id my-project
```

### 4. RAG 기반 프롬프트 개선

```bash
# 프롬프트 개선
python cursor_rag_client.py enhance "사용자 로그인 기능을 구현해주세요" --project-id my-project

# 코드 생성
python cursor_rag_client.py generate "React 컴포넌트로 사용자 프로필 페이지를 만들어주세요" --project-id my-project

# 검색 및 요약
python cursor_rag_client.py search "사용자 인증은 어떻게 구현되어 있나요?" --project-id my-project
```

## 🔧 API 엔드포인트

### RAG 기반 엔드포인트

- `POST /api/v1/rag/enhance-prompt` - 프롬프트 개선
- `POST /api/v1/rag/generate-code` - 코드 생성
- `POST /api/v1/rag/search-summarize` - 검색 및 요약

### 파일 감시 엔드포인트

- ~~`POST /api/v1/watcher/start` - 파일 감시 시작~~ (더 이상 사용되지 않음)
- ~~`POST /api/v1/watcher/stop` - 파일 감시 중지~~ (더 이상 사용되지 않음)
- ~~`GET /api/v1/watcher/status` - 감시 상태 조회~~ (더 이상 사용되지 않음)

**참고**: 파일 감시 기능은 이제 클라이언트에서 로컬로 실행됩니다.

### 기존 엔드포인트

- `POST /api/v1/upload-files` - 개별 파일 업로드
- `POST /api/v1/upload-batch` - 배치 파일 업로드
- `GET /api/v1/heartbeat` - 헬스체크

## 🎯 워크플로우

### 개발 워크플로우

```bash
# 1. 프로젝트 초기 설정
python cursor_rag_client.py init
python cursor_rag_client.py upload --project-path ./my-project --project-id my-project

# 2. 파일 감시 시작 (백그라운드)
python cursor_rag_client.py watch --project-path ./my-project --project-id my-project --keep-alive

# 3. 코딩 시작 - 파일 변경 시 자동 업로드됨
# (Cursor에서 코드 작성)

# 4. 프롬프트 개선 활용
python cursor_rag_client.py enhance "새로운 API 엔드포인트를 추가해주세요" --project-id my-project

# 5. 생성된 향상된 프롬프트를 Cursor에 붙여넣기
```

### 프롬프트 포맷

```text
# Context from my project
## src/components/UserProfile.jsx
**Type:** auto_uploaded_file
**Content:**
import React from 'react';
const UserProfile = ({ user }) => {
  return (
    <div className="user-profile">
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
};

## src/api/auth.js
**Type:** auto_uploaded_file
**Content:**
export const login = async (credentials) => {
  const response = await fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials)
  });
  return response.json();
};

---
Based on the context above, please write code to satisfy the following requirement.

Requirement: 새로운 API 엔드포인트를 추가해주세요

Please provide:
1. A complete, working code solution
2. Brief explanation of the implementation
3. Any necessary imports or dependencies
4. Usage examples if applicable

Focus on using the project context to maintain consistency with existing code patterns, naming conventions, and architecture.
```

## 📊 성능 최적화

### 파일 감시 최적화

- **디바운스**: 2초 디바운스로 중복 이벤트 방지
- **해시 기반 중복 감지**: 파일 내용 변경 시에만 업로드
- **지능형 필터링**: 34개 무시 디렉토리, 16개 무시 파일 패턴
- **확장자 필터링**: 40개 이상의 코드 파일 확장자 지원

### 벡터 검색 최적화

- **임베딩 모델**: text-embedding-3-small (빠른 속도)
- **청크 분할**: 1000자 청크, 200자 오버랩
- **컨텍스트 제한**: 기본 5개 컨텍스트 (설정 가능)

## 🔍 체크리스트

### 구현 검증 체크리스트

- [x] **MCP SSE 서버** - FastMCP 기반 SSE 통신 구현
- [x] **RAG 파이프라인** - LangChain 기반 프롬프트 재구성
- [x] **Chroma 검색** - 의미 기반 문서 검색 기능
- [x] **프롬프트 전달** - 향상된 프롬프트 Cursor 전달
- [x] **파일 감시** - 실시간 파일 변경 감지
- [x] **자동 업로드** - 변경된 파일 자동 벡터 저장

### 테스트 시나리오

1. ✅ **파일 업로드 테스트**: 프로젝트 파일들이 정상적으로 업로드되는지 확인
2. ✅ **파일 감시 테스트**: 파일 변경 시 자동 업로드되는지 확인
3. ✅ **프롬프트 개선 테스트**: 관련 컨텍스트가 포함되어 프롬프트가 향상되는지 확인
4. ✅ **코드 생성 테스트**: 프로젝트 컨텍스트를 활용한 코드가 생성되는지 확인
5. ✅ **검색 기능 테스트**: 의미 기반 검색이 정확한 결과를 반환하는지 확인

## 🐛 문제 해결

### 일반적인 문제

1. **서버 연결 실패**

   ```bash
   # 서버 상태 확인
   python cursor_rag_client.py health

   # Docker 컨테이너 상태 확인
   docker-compose ps
   ```

2. **파일 업로드 실패**

   ```bash
   # 로그 확인
   tail -f data/logs/mcp_server.log

   # 권한 확인
   ls -la data/chroma/
   ```

3. **프롬프트 개선 실패**

   ```bash
   # OpenAI API 키 확인
   echo $OPENAI_API_KEY

   # 벡터 DB 상태 확인
   python cursor_rag_client.py search "test" --project-id my-project
   ```

### 로그 위치

- **MCP 서버 로그**: `data/logs/mcp_server.log`
- **에러 로그**: `data/logs/mcp_server_error.log`
- **Chroma DB**: `data/chroma/`

## 🚀 고급 사용법

### 1. 프로젝트별 설정

```json
{
  "mcp_server_url": "http://localhost:8000",
  "project_id": "lovechedule-app",
  "auto_watch": true,
  "context_limit": 10,
  "ignore_patterns": [
    "node_modules",
    ".git",
    "__pycache__",
    "dist",
    "build",
    "coverage"
  ]
}
```

### 2. 배치 처리

```bash
# 여러 프로젝트 동시 감시
python cursor_rag_client.py watch --project-path ./frontend --project-id frontend
python cursor_rag_client.py watch --project-path ./backend --project-id backend
```

### 3. 커스텀 프롬프트 템플릿

```python
# services/langchain_rag_service.py에서 템플릿 수정
template = """# Context from my project
{context}

---
Based on the context above, please write code to satisfy the following requirement.

Requirement: {question}

Please provide:
1. A complete, working code solution
2. Brief explanation of the implementation
3. Any necessary imports or dependencies
4. Usage examples if applicable

Focus on using the project context to maintain consistency with existing code patterns, naming conventions, and architecture."""
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 라이선스

MIT License

## 🔗 관련 링크

- [FastMCP](https://github.com/jlowin/fastmcp)
- [LangChain](https://github.com/hwchase17/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [Cursor](https://cursor.sh/)

---

**Made with ❤️ by Cursor RAG Team**
