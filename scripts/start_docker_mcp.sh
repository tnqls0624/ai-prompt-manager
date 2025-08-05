#!/bin/bash

echo "=== FastMCP Docker 서버 시작 ==="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker와 Docker Compose 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker가 설치되지 않았습니다${NC}"
    echo "Docker를 설치하세요: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Error: Docker Compose가 설치되지 않았습니다${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker $(docker --version | cut -d' ' -f3) 확인됨${NC}"

# 임베딩 모델 타입 확인
EMBEDDING_MODEL_TYPE_ENV=""
if [ -f ".env" ]; then
    # .env 파일에서 EMBEDDING_MODEL_TYPE 확인
    EMBEDDING_MODEL_TYPE_ENV=$(grep "^EMBEDDING_MODEL_TYPE=" .env 2>/dev/null | cut -d'=' -f2)
fi

# 환경변수 또는 .env 파일에서 확인
if [ "$EMBEDDING_MODEL_TYPE" = "deepseek" ] || [ "$EMBEDDING_MODEL_TYPE_ENV" = "deepseek" ]; then
    echo -e "${GREEN}✅ DeepSeek 모델 사용 - OpenAI API 키 불필요${NC}"
else
    # OpenAI API 키 확인
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  Warning: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다${NC}"
        echo "다음 명령으로 설정하세요:"
        echo "  export OPENAI_API_KEY=your_openai_api_key_here"
        echo ""
        echo -e "${BLUE}💡 또는 DeepSeek 모델을 사용하려면:${NC}"
        echo "  1. .env 파일에 EMBEDDING_MODEL_TYPE=deepseek 추가"
        echo "  2. ./scripts/setup_deepseek_r1.sh 실행"
        echo ""
        read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✅ OpenAI API 키 확인됨${NC}"
    fi
fi

# 기존 컨테이너 정리
echo -e "${BLUE}🧹 기존 컨테이너 정리 중...${NC}"

# Docker 데몬 상태 확인
if ! docker info &>/dev/null; then
    echo -e "${RED}❌ Docker 데몬이 실행되지 않았습니다${NC}"
    echo "Docker Desktop을 시작하거나 Docker 서비스를 실행하세요"
    exit 1
fi

# 기존 컨테이너 확인 및 정리
echo -e "${YELLOW}📋 기존 컨테이너 확인 중...${NC}"
EXISTING_CONTAINERS=$(docker ps -a --filter "name=fastmcp-prompt-enhancement" --filter "name=chromadb-server" --filter "name=deepseek-r1-server" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || true)

if [ -n "$EXISTING_CONTAINERS" ]; then
    echo "기존 컨테이너 발견:"
    echo "$EXISTING_CONTAINERS"
    echo ""
    
    # 강제 종료 및 제거
    echo -e "${YELLOW}🛑 컨테이너 강제 종료 중...${NC}"
    docker stop fastmcp-prompt-enhancement chromadb-server deepseek-r1-server 2>/dev/null || true
    
    echo -e "${YELLOW}🗑️  컨테이너 제거 중...${NC}"
    docker rm fastmcp-prompt-enhancement chromadb-server deepseek-r1-server 2>/dev/null || true
    
    # Docker Compose 정리 (타임아웃 설정)
    echo -e "${YELLOW}🔧 Docker Compose 정리 중...${NC}"
    timeout 30 docker-compose down 2>/dev/null || timeout 30 docker compose down 2>/dev/null || true
    
    echo -e "${GREEN}✅ 컨테이너 정리 완료${NC}"
else
    echo -e "${GREEN}✅ 기존 컨테이너 없음${NC}"
fi

# 📁 필요한 디렉토리들 생성
echo -e "${BLUE}📁 데이터 디렉토리 생성 중...${NC}"
mkdir -p ./data/chroma
mkdir -p ./data/logs
mkdir -p ./backups/chroma

echo "✅ 디렉토리 생성 완료:"
echo "   - ./data/chroma (ChromaDB 데이터)"  
echo "   - ./data/logs (애플리케이션 로그)"
echo "   - ./backups/chroma (백업 파일)"

# Docker 이미지 빌드 및 컨테이너 시작
echo -e "${BLUE}🔨 Docker 이미지 빌드 중...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose build
else
    docker compose build
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: Docker 이미지 빌드 실패${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 FastMCP 서버 시작 중...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: 컨테이너 시작 실패${NC}"
    exit 1
fi

echo -e "${GREEN}✅ FastMCP 서버와 ChromaDB가 Docker에서 시작되었습니다!${NC}"
echo ""

# 컨테이너 상태 확인
echo -e "${BLUE}📊 컨테이너 상태:${NC}"
docker ps --filter "name=fastmcp-prompt-enhancement"
docker ps --filter "name=chromadb-server"

echo ""
echo -e "${YELLOW}🔧 Cursor에서 사용하려면 다음 설정을 사용하세요:${NC}"
echo ""
echo -e "${BLUE}방법 1: Docker exec 방식${NC}"
cat << 'EOF'
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "docker",
      "args": ["exec", "-i", "fastmcp-prompt-enhancement", "python", "mcp_server.py"]
    }
  }
}
EOF

echo ""
echo -e "${BLUE}방법 2: Docker run 방식 (매번 새 컨테이너)${NC}"
cat << 'EOF'
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-v", "$(pwd)/chroma_db:/app/chroma_db",
               "-e", "OPENAI_API_KEY=$OPENAI_API_KEY",
               "mcp-server_fastmcp-server", "python", "mcp_server.py"]
    }
  }
}
EOF

echo ""
echo -e "${YELLOW}📋 유용한 명령어:${NC}"
echo "  FastMCP 로그:  docker logs fastmcp-prompt-enhancement -f"
echo "  ChromaDB 로그: docker logs chromadb-server -f"
echo "  컨테이너 접속: docker exec -it fastmcp-prompt-enhancement bash"
echo "  ChromaDB UI:   http://localhost:8001 (웹 인터페이스)"
echo "  서버 중지:     ./scripts/stop_docker_mcp.sh"
echo "  상태 확인:     docker ps"
echo ""
echo -e "${YELLOW}🔍 서비스 정보:${NC}"
echo "  FastMCP 서버:  http://localhost:8000"
echo "  ChromaDB API:  http://localhost:8001"
echo ""
echo -e "${GREEN}🎉 Docker 기반 FastMCP + ChromaDB 서버 준비 완료!${NC}" 