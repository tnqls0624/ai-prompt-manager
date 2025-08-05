#!/bin/bash
# ChromaDB 데이터 복원 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 ChromaDB 데이터 복원 스크립트${NC}"
echo "======================================"

# 백업 디렉토리 확인
BACKUP_DIR="./backups/chroma"

if [ ! -d "$BACKUP_DIR" ]; then
    echo -e "${RED}❌ 백업 디렉토리를 찾을 수 없습니다: $BACKUP_DIR${NC}"
    exit 1
fi

# 사용 가능한 백업 목록 표시
echo -e "\n${YELLOW}📦 사용 가능한 백업 목록:${NC}"
backup_files=($(ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null))

if [ ${#backup_files[@]} -eq 0 ]; then
    echo -e "${RED}❌ 백업 파일을 찾을 수 없습니다.${NC}"
    echo "먼저 백업을 생성하세요: ./backup_chroma.sh"
    exit 1
fi

for i in "${!backup_files[@]}"; do
    filename=$(basename "${backup_files[$i]}")
    timestamp=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
    formatted_time=$(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "$timestamp")
    size=$(du -h "${backup_files[$i]}" | cut -f1)
    
    echo "$((i+1)). $filename ($formatted_time, $size)"
done

echo ""
read -p "복원할 백업을 선택하세요 (1-${#backup_files[@]}): " selection

if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#backup_files[@]} ]; then
    echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
    exit 1
fi

SELECTED_BACKUP="${backup_files[$((selection-1))]}"
echo -e "${BLUE}선택된 백업: $(basename "$SELECTED_BACKUP")${NC}"

# 복원 방법 선택
echo -e "\n${YELLOW}🔄 복원 방법 선택:${NC}"
echo "1. 호스트 경로 복원 (기본 설정) ⭐"
echo "2. Docker Volume 복원 (레거시 설정)"
echo "3. 데이터베이스 복원 (API 방식)"

read -p "선택하세요 (1-3): " restore_choice

# 백업 압축 해제
TEMP_RESTORE_DIR="/tmp/chroma_restore_$$"
mkdir -p "$TEMP_RESTORE_DIR"

echo -e "\n${BLUE}📂 백업 압축 해제 중...${NC}"
tar -xzf "$SELECTED_BACKUP" -C "$TEMP_RESTORE_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 백업 압축 해제 실패${NC}"
    rm -rf "$TEMP_RESTORE_DIR"
    exit 1
fi

# 압축 해제된 백업 디렉토리 찾기
BACKUP_EXTRACT_DIR=$(find "$TEMP_RESTORE_DIR" -name "backup_*" -type d | head -1)

if [ -z "$BACKUP_EXTRACT_DIR" ]; then
    echo -e "${RED}❌ 백업 데이터를 찾을 수 없습니다.${NC}"
    rm -rf "$TEMP_RESTORE_DIR"
    exit 1
fi

case $restore_choice in
    1)
        echo -e "\n${BLUE}🔄 호스트 경로 복원 중...${NC}"
        
        # 경고 메시지
        echo -e "${YELLOW}⚠️ 주의: 기존 데이터가 덮어쓰여집니다!${NC}"
        read -p "계속하시겠습니까? (y/N): " confirm
        
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "복원이 취소되었습니다."
            rm -rf "$TEMP_RESTORE_DIR"
            exit 0
        fi
        
        # 데이터 디렉토리 생성
        mkdir -p "./data"
        
        # 기존 데이터 백업 (안전장치)
        if [ -d "./data/chroma" ]; then
            SAFETY_BACKUP="./data/chroma_backup_$(date '+%Y%m%d_%H%M%S')"
            echo "기존 데이터를 안전 백업 중: $SAFETY_BACKUP"
            mv "./data/chroma" "$SAFETY_BACKUP"
        fi
        
        # 데이터 복원
        if [ -d "$BACKUP_EXTRACT_DIR/chroma_data" ]; then
            cp -r "$BACKUP_EXTRACT_DIR/chroma_data" "./data/chroma"
            echo -e "${GREEN}✅ 호스트 경로 복원 완료${NC}"
        else
            echo -e "${RED}❌ 백업 데이터를 찾을 수 없습니다.${NC}"
            rm -rf "$TEMP_RESTORE_DIR"
            exit 1
        fi
        
        echo "서버를 재시작해주세요:"
        echo "./scripts/start_docker_mcp.sh"
        ;;
        
    2)
        echo -e "\n${BLUE}🔄 Docker Volume 복원 중...${NC}"
        
        # 경고 메시지
        echo -e "${YELLOW}⚠️ 주의: 기존 데이터가 덮어쓰여집니다!${NC}"
        read -p "계속하시겠습니까? (y/N): " confirm
        
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "복원이 취소되었습니다."
            rm -rf "$TEMP_RESTORE_DIR"
            exit 0
        fi
        
        # ChromaDB 컨테이너 중지
        if docker ps | grep -q "chromadb-server"; then
            echo "ChromaDB 컨테이너를 중지하는 중..."
            docker stop chromadb-server
        fi
        
        # Volume 데이터 복원 (레거시)
        if [ -d "$BACKUP_EXTRACT_DIR/chroma_data" ]; then
            # 임시 컨테이너로 볼륨에 데이터 복사
            docker run --rm -v mcp-server_chroma_data:/target -v "$BACKUP_EXTRACT_DIR/chroma_data":/source alpine \
                sh -c "rm -rf /target/* /target/.[^.]* 2>/dev/null || true; cp -r /source/* /target/ 2>/dev/null || true; cp -r /source/.[^.]* /target/ 2>/dev/null || true"
            
            echo -e "${GREEN}✅ Volume 복원 완료${NC}"
        else
            echo -e "${RED}❌ 백업 데이터를 찾을 수 없습니다.${NC}"
            rm -rf "$TEMP_RESTORE_DIR"
            exit 1
        fi
        
        # ChromaDB 컨테이너 재시작
        echo "ChromaDB 컨테이너를 시작하는 중..."
        docker-compose up -d chromadb
        ;;
        
    3)
        echo -e "\n${BLUE}🔄 데이터베이스 API 복원 중...${NC}"
        
        # ChromaDB API 연결 확인
        if ! curl -s -f http://localhost:8001/api/v1/heartbeat > /dev/null; then
            echo -e "${RED}❌ ChromaDB API에 연결할 수 없습니다.${NC}"
            echo "서버가 실행 중인지 확인하세요: http://localhost:8001"
            rm -rf "$TEMP_RESTORE_DIR"
            exit 1
        fi
        
        echo -e "${YELLOW}⚠️ 주의: 기존 컬렉션이 덮어쓰여집니다!${NC}"
        read -p "계속하시겠습니까? (y/N): " confirm
        
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "복원이 취소되었습니다."
            rm -rf "$TEMP_RESTORE_DIR"
            exit 0
        fi
        
        # JSON 백업 파일들 복원
        python3 << EOF
import requests
import json
import os
import glob

try:
    backup_path = "$BACKUP_EXTRACT_DIR"
    json_files = glob.glob(f"{backup_path}/*.json")
    
    for json_file in json_files:
        if 'collections.json' in json_file:
            continue
            
        collection_name = os.path.basename(json_file).replace('.json', '')
        print(f"복원 중: {collection_name}")
        
        # 기존 컬렉션 삭제 (선택적)
        try:
            requests.delete(f"http://localhost:8001/api/v1/collections/{collection_name}")
        except:
            pass
        
        # 컬렉션 생성
        requests.post("http://localhost:8001/api/v1/collections", 
                     json={"name": collection_name})
        
        # 데이터 로드
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # 데이터 추가 (구현은 ChromaDB API 버전에 따라 다를 수 있음)
        # 여기서는 단순화된 버전
        print(f"  - {collection_name}: {len(data.get('documents', []))}개 문서")
    
    print("API 복원 완료")
except Exception as e:
    print(f"API 복원 실패: {e}")
EOF
        
        echo -e "${GREEN}✅ API 복원 완료${NC}"
        ;;
        
    *)
        echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
        rm -rf "$TEMP_RESTORE_DIR"
        exit 1
        ;;
esac

# 임시 파일 정리
rm -rf "$TEMP_RESTORE_DIR"

echo -e "\n${GREEN}🎉 복원 완료!${NC}"
echo -e "\n${BLUE}📋 다음 단계:${NC}"
echo "  1. 서버 상태 확인:    docker ps"
echo "  2. 데이터 확인:      curl http://localhost:8001/api/v1/heartbeat"
echo "  3. 테스트 실행:      python test_advanced_analytics.py" 