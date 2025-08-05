# 🚀 DeepSeek R1 로컬 임베딩 설정

OpenAI API 비용 없이 로컬에서 DeepSeek R1 모델을 사용하여 임베딩을 생성합니다.

## 📋 설정 방법

### 1. 환경변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# .env 파일 생성
cat > .env << 'EOF'
# 🤖 임베딩 모델 설정
EMBEDDING_MODEL_TYPE=deepseek

# 🚀 DeepSeek R1 모델 설정
DEEPSEEK_MODEL_NAME=r1-1776:latest

# 📝 로깅 설정
LOG_LEVEL=INFO

# 📊 분석 설정
ENABLE_ADVANCED_ANALYTICS=true
CLUSTERING_ALGORITHM=kmeans
MAX_CLUSTERS=10

# 🧠 AI 설정
MAX_CONTEXT_LENGTH=5
SIMILARITY_THRESHOLD=0.7
EOF
```

### 2. Docker 서비스 시작

```bash
# Docker 서비스 시작
docker-compose up -d

# 서비스 상태 확인
docker ps | grep "deepseek\|fastmcp\|chromadb"
```

### 3. DeepSeek R1 모델 설치

```bash
# DeepSeek R1 모델 설치 스크립트 실행
chmod +x scripts/setup_deepseek_r1.sh
./scripts/setup_deepseek_r1.sh
```

### 4. 서비스 재시작

```bash
# FastMCP 서버 재시작하여 새 설정 적용
docker-compose restart fastmcp-server
```

## 🔧 모델 설정

### 사용 가능한 모델들

1. **r1-1776:latest** - 실제 DeepSeek R1 모델 (추천, 42GB)
2. **deepseek-coder:7b** - 코딩 특화 모델 (7B 파라미터)
3. **deepseek-coder:33b** - 더 큰 모델 (GPU 필요, 33B 파라미터)

### 모델 변경 방법

```bash
# 다른 모델 설치
docker exec deepseek-r1-server ollama pull deepseek-coder:33b

# .env 파일에서 모델명 변경
DEEPSEEK_MODEL_NAME=deepseek-coder:33b

# 서비스 재시작
docker-compose restart fastmcp-server
```

## 🖥️ GPU 지원 (선택사항)

GPU가 있다면 `docker-compose.yml`에서 GPU 지원을 활성화하세요:

```yaml
# docker-compose.yml에서 주석 해제
deepseek-r1:
  # ... 기타 설정 ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## 📊 성능 비교

| 모델        | 비용    | 속도    | 품질    | 프라이버시 |
| ----------- | ------- | ------- | ------- | ---------- |
| OpenAI API  | 💰 유료 | ⚡ 빠름 | 🌟 높음 | ⚠️ 외부    |
| DeepSeek R1 | 🆓 무료 | 🐢 느림 | ⭐ 중간 | 🔒 로컬    |

## 🔍 문제 해결

### 모델 다운로드 실패

```bash
# 컨테이너 로그 확인
docker logs deepseek-r1-server -f

# 수동으로 모델 다운로드
docker exec -it deepseek-r1-server ollama pull deepseek-coder:7b
```

### 임베딩 생성 실패

```bash
# API 엔드포인트 확인
curl http://localhost:11434/api/tags

# 서비스 상태 확인
docker exec deepseek-r1-server ollama list
```

### 메모리 부족

```bash
# 더 작은 모델 사용
DEEPSEEK_MODEL_NAME=llama2:7b

# 또는 Docker 메모리 제한 설정
# docker-compose.yml에서 메모리 제한 추가
```

## 💡 팁

1. **첫 실행 시**: 모델 다운로드로 인해 시간이 오래 걸릴 수 있습니다
2. **GPU 권장**: 성능 향상을 위해 GPU 사용을 권장합니다
3. **모델 크기**: 메모리에 맞는 모델 크기를 선택하세요
4. **로컬 우선**: 개인정보 보호를 위해 로컬 모델을 우선 사용하세요

## 🚀 확인 방법

```bash
# 임베딩 서비스 테스트
docker exec fastmcp-prompt-enhancement python -c "
import asyncio
from services.vector_service import VectorService

async def test_embedding():
    vs = VectorService()
    if vs.embeddings:
        print('✅ 임베딩 서비스 초기화 완료')
        print(f'📊 모델 타입: {vs.embedding_model_type}')
    else:
        print('❌ 임베딩 서비스 초기화 실패')

asyncio.run(test_embedding())
"
```

이제 OpenAI API 없이도 로컬에서 임베딩을 생성할 수 있습니다! 🎉
