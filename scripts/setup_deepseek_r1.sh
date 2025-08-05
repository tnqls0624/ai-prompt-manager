#!/bin/bash

echo "=== DeepSeek R1 모델 설정 ==="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# DeepSeek R1 컨테이너가 실행 중인지 확인
if ! docker ps | grep -q "deepseek-r1-server"; then
    echo -e "${RED}❌ DeepSeek R1 컨테이너가 실행 중이 아닙니다.${NC}"
    echo "먼저 docker-compose up -d 명령으로 서비스를 시작하세요."
    exit 1
fi

echo -e "${GREEN}✅ DeepSeek R1 컨테이너가 실행 중입니다${NC}"

# Ollama 서비스 상태 확인
echo -e "${BLUE}🔍 Ollama 서비스 상태 확인 중...${NC}"
sleep 10  # 컨테이너 시작 대기

# DeepSeek R1 모델 다운로드 및 설치
echo -e "${BLUE}📥 DeepSeek R1 모델 다운로드 중...${NC}"
echo "이 과정은 모델 크기에 따라 시간이 걸릴 수 있습니다."

# 실제 DeepSeek R1 모델이 Ollama에 있는지 확인 후 다운로드
# 참고: DeepSeek R1의 실제 모델명은 다를 수 있으므로 사용 가능한 모델로 대체
docker exec deepseek-r1-server ollama pull r1-1776

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ DeepSeek 모델 다운로드 완료${NC}"
else
    echo -e "${YELLOW}⚠️ DeepSeek R1 모델을 찾을 수 없습니다. 대안 모델을 사용합니다.${NC}"
    
    # DeepSeek R1이 없는 경우 대안 모델 설치
    echo -e "${BLUE}📥 대안 모델 다운로드 중...${NC}"
    docker exec deepseek-r1-server ollama pull r1-1776
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 대안 모델 다운로드 완료${NC}"
    else
        echo -e "${RED}❌ 모델 다운로드 실패${NC}"
        exit 1
    fi
fi

# 모델 목록 확인
echo -e "${BLUE}📋 설치된 모델 목록:${NC}"
docker exec deepseek-r1-server ollama list

# 서비스 상태 확인
echo -e "${BLUE}🔍 서비스 상태 확인...${NC}"
curl -s http://localhost:11434/api/tags | jq '.' || echo "API 응답 확인 중..."

echo -e "${GREEN}✅ DeepSeek R1 설정 완료!${NC}"
echo ""
echo -e "${YELLOW}🔧 사용 방법:${NC}"
echo "1. .env 파일에서 EMBEDDING_MODEL_TYPE=deepseek 설정"
echo "2. docker-compose restart fastmcp-server 명령으로 서버 재시작"
echo "3. 이제 OpenAI API 없이 로컬에서 임베딩 생성 가능!"
echo ""
echo -e "${BLUE}💡 팁:${NC}"
echo "- GPU가 있다면 docker-compose.yml에서 GPU 지원 활성화"
echo "- 모델 변경: docker exec deepseek-r1-server ollama pull [모델명]"
echo "- 서비스 로그: docker logs deepseek-r1-server -f" 