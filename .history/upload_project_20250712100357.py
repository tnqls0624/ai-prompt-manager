#!/usr/bin/env python3
"""
프로젝트 파일을 MCP 서버로 업로드하고 벡터 인덱싱하는 스크립트
"""

import asyncio
import aiohttp
import aiofiles
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor
import math

# 지원하는 파일 확장자들
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs',
    '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
    '.md', '.txt', '.rst', '.asciidoc',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
    '.sql', '.sh', '.bash', '.ps1',
    '.html', '.css', '.scss', '.sass', '.less',
    '.vue', '.svelte', '.astro'
}

# 무시할 디렉토리들
IGNORE_DIRECTORIES = {
    # JavaScript/Node.js 관련
    'node_modules', 'bower_components', 'jspm_packages', 'typings',
    
    # Python 관련
    '__pycache__', '.pytest_cache', '.mypy_cache', 'venv', 'env', '.env',
    
    # 버전 관리 시스템
    '.git', '.svn', '.hg',
    
    # IDE 관련
    '.vscode', '.idea',
    
    # 빌드 관련
    'dist', 'build', 'target', 'out', '.next', 'bin', 'obj',
    
    # 컴파일러별 빌드 디렉토리
    'Debug', 'Release',
    
    # 언어별 패키지 관리
    'vendor', 'pkg',
    
    # 캐시 및 임시 파일
    'cache', 'tmp', 'temp', 'coverage', 'logs',
    
    # 정적 파일 및 에셋
    'assets', 'public', 'static',
    
    # 데이터베이스 관련
    'chroma_db'
}

# 무시할 파일들
IGNORE_FILES = {
    # 환경 및 설정 파일
    '.gitignore', '.dockerignore', '.env', '.env.local',
    
    # 패키지 매니저 락 파일들
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
    'poetry.lock', 'Pipfile.lock', 'pdm.lock',
    'composer.lock', 'Gemfile.lock', 'Cargo.lock', 
    'go.sum', 'mix.lock', 'pubspec.lock'
}

class FastProjectUploader:
    """고성능 프로젝트 업로드 클래스"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.upload_url = f"{self.server_url}/api/v1/upload-batch"
        self.max_workers = 20  # 병렬 파일 읽기 워커 수
        self.batch_size = 200  # 배치 크기 (더 큰 단위로)
        
    async def read_file_async(self, file_path: Path, project_path: Path) -> Dict[str, Any]:
        """비동기 파일 읽기"""
        try:
            # 파일 크기 체크 (10MB 이상은 제외)
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return None
            
            # 비동기 파일 읽기
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            # 너무 작은 파일 제외
            if len(content.strip()) < 10:
                return None
            
            relative_path = file_path.relative_to(project_path)
            return {
                "path": str(relative_path),
                "content": content,
                "size": len(content)
            }
            
        except Exception as e:
            print(f"⚠️  파일 읽기 실패 {file_path}: {e}")
            return None
    
    async def scan_project_files_async(self, project_path: Path) -> List[Dict[str, Any]]:
        """비동기 병렬 파일 스캔"""
        print(f"🔍 프로젝트 스캔 중: {project_path}")
        
        # 먼저 모든 파일 경로를 수집
        file_paths = []
        for root, dirs, files in os.walk(project_path):
            # 무시할 디렉토리 제거
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRECTORIES]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                # .으로 시작하는 숨김 파일 제외
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # 지원하는 확장자만 처리
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_paths.append(file_path)
        
        print(f"📂 {len(file_paths)}개 파일 발견, 병렬 읽기 시작...")
        
        # 파일들을 비동기 병렬로 읽기
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def read_with_semaphore(file_path):
            async with semaphore:
                return await self.read_file_async(file_path, project_path)
        
        # 모든 파일을 병렬로 읽기
        results = await asyncio.gather(
            *[read_with_semaphore(file_path) for file_path in file_paths],
            return_exceptions=True
        )
        
        # 유효한 파일만 필터링
        files_data = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                files_data.append(result)
        
        print(f"✅ {len(files_data)}개 파일 읽기 완료")
        return files_data
    
    async def upload_batch(self, files_batch: List[Dict[str, Any]], project_id: str, project_name: str, batch_num: int, total_batches: int) -> Dict[str, Any]:
        """배치 업로드"""
        upload_data = {
            "project_id": project_id,
            "project_name": project_name,
            "files": files_batch
        }
        
        print(f"📤 배치 {batch_num}/{total_batches} 업로드 중... ({len(files_batch)}개 파일)")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.upload_url,
                json=upload_data,
                timeout=aiohttp.ClientTimeout(total=600)  # 10분 타임아웃
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 배치 {batch_num}/{total_batches} 완료 (성공률: {result.get('success_rate', 0)}%)")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"배치 {batch_num} 업로드 실패 (HTTP {response.status}): {error_text}")
    
    async def upload_project(self, project_path: str, project_id: str, project_name: str = None) -> Dict[str, Any]:
        """고성능 프로젝트 업로드"""
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise FileNotFoundError(f"프로젝트 경로가 존재하지 않습니다: {project_path}")
        
        if not project_name:
            project_name = project_path.name
        
        # 비동기 병렬 파일 스캔
        files_data = await self.scan_project_files_async(project_path)
        
        if not files_data:
            raise ValueError("업로드할 파일이 없습니다.")
        
        # 파일 크기별 통계
        total_size = sum(f["size"] for f in files_data)
        print(f"📊 총 크기: {total_size / 1024:.1f} KB")
        
        # 배치로 나누기
        total_batches = math.ceil(len(files_data) / self.batch_size)
        print(f"🚀 {total_batches}개 배치로 나누어 병렬 업로드 시작...")
        
        start_time = time.time()
        
        # 배치들을 병렬로 업로드
        upload_tasks = []
        for i in range(0, len(files_data), self.batch_size):
            batch = files_data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            task = self.upload_batch(batch, project_id, project_name, batch_num, total_batches)
            upload_tasks.append(task)
        
        # 모든 배치 업로드를 병렬로 실행
        batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # 결과 집계
        total_received = 0
        total_indexed = 0
        total_failed = 0
        tech_stacks = set()
        
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"❌ 배치 업로드 실패: {result}")
                continue
            
            total_received += result.get('total_files_received', 0)
            total_indexed += result.get('indexed_files_count', 0)
            total_failed += result.get('failed_files_count', 0)
            tech_stacks.update(result.get('tech_stack', []))
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        success_rate = (total_indexed / total_received * 100) if total_received > 0 else 0
        
        print(f"🎉 모든 배치 업로드 완료! ({upload_time:.2f}초)")
        print(f"   📤 전송된 파일: {total_received}개")
        print(f"   ✅ 인덱싱된 파일: {total_indexed}개") 
        print(f"   ❌ 실패한 파일: {total_failed}개")
        print(f"   📈 성공률: {success_rate:.1f}%")
        print(f"   🚀 처리 속도: {total_indexed / upload_time:.1f} 파일/초")
        print(f"   🔧 기술 스택: {', '.join(sorted(tech_stacks))}")
        
        return {
            "success": True,
            "total_files_received": total_received,
            "indexed_files_count": total_indexed,
            "failed_files_count": total_failed,
            "success_rate": success_rate,
            "upload_time": upload_time,
            "processing_speed": total_indexed / upload_time if upload_time > 0 else 0,
            "tech_stack": sorted(list(tech_stacks))
        }

# 기존 ProjectUploader는 호환성을 위해 유지
class ProjectUploader(FastProjectUploader):
    """기존 ProjectUploader 호환성 유지"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        super().__init__(server_url)
        self.max_workers = 10  # 기존 설정
        self.batch_size = 100  # 기존 설정
    
    def scan_project_files(self, project_path: Path) -> List[Dict[str, Any]]:
        """동기 파일 스캔 (기존 호환성)"""
        return asyncio.run(self.scan_project_files_async(project_path))

async def main():
    """메인 함수 - 고성능 업로드 사용"""
    print("🚀 고성능 프로젝트 업로드 도구")
    print("=" * 50)
    
    # 설정
    PROJECT_PATH = "/Users/soobeen/Desktop/Project/lovechedule"
    PROJECT_ID = "lovechedule"
    PROJECT_NAME = "LoveSchedule App"
    SERVER_URL = "http://localhost:8000"
    
    # 고성능 업로더 사용
    uploader = FastProjectUploader(SERVER_URL)
    
    try:
        # 서버 연결 확인
        print(f"🔗 서버 연결 확인: {SERVER_URL}")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/api/v1/health") as response:
                if response.status == 200:
                    print("✅ 서버 상태: healthy")
                else:
                    print(f"❌ 서버 연결 실패 (HTTP {response.status})")
                    return
        
        print()
        print(f"📂 프로젝트: {PROJECT_PATH}")
        print(f"🏷️  프로젝트 ID: {PROJECT_ID}")
        print(f"📝 프로젝트명: {PROJECT_NAME}")
        
        # 고성능 업로드 실행
        result = await uploader.upload_project(PROJECT_PATH, PROJECT_ID, PROJECT_NAME)
        
        print("\n🎉 업로드 완료!")
        print("이제 MCP 서버에서 다음 명령으로 검색할 수 있습니다:")
        print(f"  search_project_files(query='your search term', project_id='{PROJECT_ID}')")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return

if __name__ == "__main__":
    # aiofiles 의존성 확인
    try:
        import aiofiles
    except ImportError:
        print("❌ aiofiles 라이브러리가 필요합니다.")
        print("설치 명령: pip install aiofiles")
        exit(1)
    
    # 명령행 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description="고성능 프로젝트 업로드 도구")
    parser.add_argument("--project-path", default="/Users/soobeen/Desktop/Project/lovechedule", help="프로젝트 경로")
    parser.add_argument("--project-id", default="lovechedule", help="프로젝트 ID")
    parser.add_argument("--project-name", default="LoveSchedule App", help="프로젝트 이름")
    parser.add_argument("--server-url", default="http://localhost:8000", help="서버 URL")
    
    args = parser.parse_args()
    
    # 전역 설정 업데이트
    import types
    main_module = types.ModuleType('__main__')
    main_module.PROJECT_PATH = args.project_path
    main_module.PROJECT_ID = args.project_id
    main_module.PROJECT_NAME = args.project_name
    main_module.SERVER_URL = args.server_url
    
    # 메인 함수 실행
    asyncio.run(main()) 