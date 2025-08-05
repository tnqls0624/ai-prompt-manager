#!/bin/bash

echo "=== FastMCP Docker 서버 정지 ==="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker가 설치되지 않았습니다${NC}"
    exit 1
fi

# 컨테이너 상태 확인
echo -e "${BLUE}📊 현재 컨테이너 상태:${NC}"
docker ps --filter "name=fastmcp-prompt-enhancement"
docker ps --filter "name=chromadb-server"

# 실행 중인 컨테이너가 있는지 확인
FASTMCP_CONTAINER_ID=$(docker ps -q --filter "name=fastmcp-prompt-enhancement")
CHROMADB_CONTAINER_ID=$(docker ps -q --filter "name=chromadb-server")

if [ -z "$FASTMCP_CONTAINER_ID" ] && [ -z "$CHROMADB_CONTAINER_ID" ]; then
    echo -e "${YELLOW}⚠️  실행 중인 FastMCP/ChromaDB 컨테이너를 찾을 수 없습니다${NC}"
    echo "다음 명령으로 모든 컨테이너를 확인하세요:"
    echo "  docker ps -a"
    exit 0
fi

echo -e "${BLUE}🛑 FastMCP + ChromaDB 서버 정지 중...${NC}"

# Docker Compose 사용해서 정지
if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    docker compose down
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ FastMCP + ChromaDB 서버가 성공적으로 정지되었습니다${NC}"
else
    echo -e "${YELLOW}⚠️  Docker Compose 정지 실패, 직접 컨테이너 정지 시도 중...${NC}"
    docker stop fastmcp-prompt-enhancement chromadb-server
    docker rm fastmcp-prompt-enhancement chromadb-server
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 모든 컨테이너가 직접 정지되었습니다${NC}"
    else
        echo -e "${RED}❌ Error: 컨테이너 정지 실패${NC}"
        exit 1
    fi
fi

# 정리 확인
echo -e "${BLUE}📊 정지 후 상태:${NC}"
docker ps --filter "name=fastmcp-prompt-enhancement"
docker ps --filter "name=chromadb-server"

# 정지된 컨테이너 확인
STOPPED_FASTMCP=$(docker ps -a -q --filter "name=fastmcp-prompt-enhancement")
STOPPED_CHROMADB=$(docker ps -a -q --filter "name=chromadb-server")
if [ -n "$STOPPED_FASTMCP" ] || [ -n "$STOPPED_CHROMADB" ]; then
    echo -e "${BLUE}🗑️  정지된 컨테이너:${NC}"
    docker ps -a --filter "name=fastmcp-prompt-enhancement"
    docker ps -a --filter "name=chromadb-server"
    echo ""
    echo -e "${YELLOW}완전히 제거하려면:${NC}"
    [ -n "$STOPPED_FASTMCP" ] && echo "  docker rm fastmcp-prompt-enhancement"
    [ -n "$STOPPED_CHROMADB" ] && echo "  docker rm chromadb-server"
fi

echo ""
echo -e "${YELLOW}📋 유용한 명령어:${NC}"
echo "  서버 재시작:     ./scripts/start_docker_mcp.sh"
echo "  이미지 정리:     docker rmi mcp-server_fastmcp-server chromadb/chroma:latest"
echo "  데이터 백업:     ./scripts/backup_chroma.sh"
echo "  데이터 복원:     ./scripts/restore_chroma.sh"
echo "  모든 정리:       docker system prune -a"
echo "  네트워크 정리:   docker network rm mcp-server_mcp-network"
echo ""
echo -e "${YELLOW}🗄️ 데이터 위치:${NC}"
echo "  ChromaDB 데이터: ./data/chroma/"
echo "  로그 파일:       ./data/logs/"
echo "  백업 파일:       ./backups/chroma/"
echo ""
echo -e "${GREEN}✅ FastMCP + ChromaDB Docker 서버 정지 완료${NC}" 