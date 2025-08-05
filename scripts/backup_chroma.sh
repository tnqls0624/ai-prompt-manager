#!/bin/bash
# ChromaDB 데이터 백업 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️ ChromaDB 데이터 백업 스크립트${NC}"
echo "======================================"

# 백업 디렉토리 생성
BACKUP_DIR="./backups/chroma"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"

mkdir -p "$BACKUP_PATH"

echo -e "\n${YELLOW}📦 백업 방법 선택:${NC}"
echo "1. 호스트 경로 백업 (기본 설정) ⭐"
echo "2. Docker Volume 백업 (레거시 설정)"
echo "3. 데이터베이스 덤프 (API 방식)"

read -p "선택하세요 (1-3): " choice

case $choice in
    1)
        echo -e "\n${BLUE}📋 호스트 경로 백업 중...${NC}"
        
        if [ -d "./data/chroma" ]; then
            cp -r ./data/chroma "$BACKUP_PATH/chroma_data"
            echo -e "${GREEN}✅ 호스트 경로 백업 완료: $BACKUP_PATH${NC}"
        else
            echo -e "${RED}❌ ./data/chroma 디렉토리를 찾을 수 없습니다.${NC}"
            echo "서버를 먼저 시작해주세요:"
            echo "./scripts/start_docker_mcp.sh"
            exit 1
        fi
        ;;
        
    2)
        echo -e "\n${BLUE}📋 Docker Volume 백업 중...${NC}"
        
        # ChromaDB 컨테이너가 실행 중인지 확인
        if docker ps | grep -q "chromadb-server"; then
            echo "ChromaDB 컨테이너가 실행 중입니다."
            
            # 컨테이너에서 데이터 복사
            docker cp chromadb-server:/chroma/chroma "$BACKUP_PATH/chroma_data"
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Volume 백업 완료: $BACKUP_PATH${NC}"
            else
                echo -e "${RED}❌ Volume 백업 실패${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}⚠️ ChromaDB 컨테이너가 실행되지 않음. Volume을 직접 백업합니다.${NC}"
            
            # Docker volume을 임시 컨테이너로 마운트하여 백업 (레거시)
            docker run --rm -v mcp-server_chroma_data:/source -v "$(pwd)/$BACKUP_PATH":/backup alpine \
                sh -c "cp -r /source/* /backup/ || cp -r /source/.[^.]* /backup/ 2>/dev/null || true"
            
            echo -e "${GREEN}✅ Volume 백업 완료: $BACKUP_PATH${NC}"
        fi
        ;;
        
    3)
        echo -e "\n${BLUE}📋 데이터베이스 덤프 중...${NC}"
        
        # ChromaDB API를 통한 데이터 덤프
        if curl -s -f http://localhost:8001/api/v1/heartbeat > /dev/null; then
            echo "ChromaDB API에 연결되었습니다."
            
            # 컬렉션 목록 가져오기
            curl -s "http://localhost:8001/api/v1/collections" > "$BACKUP_PATH/collections.json"
            
            # 각 컬렉션의 데이터 백업
            python3 << EOF
import requests
import json
import os

try:
    # 컬렉션 목록 조회
    response = requests.get("http://localhost:8001/api/v1/collections")
    collections = response.json()
    
    backup_path = "$BACKUP_PATH"
    
    for collection in collections:
        collection_name = collection['name']
        print(f"백업 중: {collection_name}")
        
        # 컬렉션 데이터 조회
        data_response = requests.post(f"http://localhost:8001/api/v1/collections/{collection_name}/query", 
                                    json={"query_texts": [""], "n_results": 1000})
        
        # 데이터 저장
        with open(f"{backup_path}/{collection_name}.json", "w") as f:
            json.dump(data_response.json(), f, indent=2)
    
    print("API 백업 완료")
except Exception as e:
    print(f"API 백업 실패: {e}")
EOF
            echo -e "${GREEN}✅ API 덤프 완료: $BACKUP_PATH${NC}"
        else
            echo -e "${RED}❌ ChromaDB API에 연결할 수 없습니다.${NC}"
            echo "서버가 실행 중인지 확인하세요: http://localhost:8001"
            exit 1
        fi
        ;;
        
    *)
        echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac

# 백업 압축
echo -e "\n${BLUE}🗜️ 백업 압축 중...${NC}"
tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "backup_${TIMESTAMP}"

if [ $? -eq 0 ]; then
    # 압축된 백업만 남기고 원본 디렉토리 삭제
    rm -rf "$BACKUP_PATH"
    
    echo -e "${GREEN}✅ 백업 완료!${NC}"
    echo -e "${YELLOW}📦 백업 파일: ${BACKUP_PATH}.tar.gz${NC}"
    echo -e "${YELLOW}📏 파일 크기: $(du -h "${BACKUP_PATH}.tar.gz" | cut -f1)${NC}"
else
    echo -e "${RED}❌ 압축 실패${NC}"
fi

echo -e "\n${BLUE}📋 백업 명령어들:${NC}"
echo "  백업 목록 보기:    ls -la ./backups/chroma/"
echo "  백업 복원하기:    ./restore_chroma.sh"
echo "  오래된 백업 정리:  find ./backups/chroma/ -name '*.tar.gz' -mtime +30 -delete" 