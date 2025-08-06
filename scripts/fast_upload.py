#!/usr/bin/env python3
"""
🚀 고성능 프로젝트 업로드 스크립트
최적화된 배치 처리와 병렬 업로드를 통한 빠른 인덱싱
"""

import asyncio
import aiohttp
import aiofiles
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import math
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 지원하는 파일 확장자들 (확장됨)
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs',
    '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.clj',
    '.md', '.txt', '.rst', '.asciidoc', '.org',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.sql', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
    '.dart', '.r', '.hs', '.elm', '.xml', '.dockerfile', '.env'
}

# 무시할 디렉토리들 (확장됨)
IGNORE_DIRECTORIES = {
    'node_modules', 'bower_components', 'jspm_packages', 'typings',
    '__pycache__', '.pytest_cache', '.mypy_cache', 'venv', 'env', '.env',
    '.git', '.svn', '.hg', '.bzr', 'CVS',
    '.vscode', '.idea', '.vs', '.vscode-test',
    'dist', 'build', 'target', 'out', '.next', 'bin', 'obj',
    'Debug', 'Release', 'vendor', 'pkg', 'cache', 'tmp', 'temp',
    'coverage', 'logs', 'assets', 'public', 'static', 'chroma_db',
    '.tox', '.nox', 'htmlcov', '.coverage', '.nyc_output',
    'lib', 'libs', 'packages', 'deps', 'external'
}

# 무시할 파일들 (확장됨)
IGNORE_FILES = {
    '.gitignore', '.dockerignore', '.env', '.env.local', '.env.production',
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'bun.lockb',
    'poetry.lock', 'Pipfile.lock', 'pdm.lock', 'requirements.txt',
    'composer.lock', 'Gemfile.lock', 'Cargo.lock', 'go.sum', 'go.mod',
    'mix.lock', 'pubspec.lock', '.DS_Store', 'Thumbs.db'
}

class FastProjectUploader:
    """고성능 프로젝트 업로더"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = None
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_workers = 50  # 병렬 파일 읽기 워커 수 증가
        self.batch_size = 300  # 배치 크기 증가
        
    async def __aenter__(self):
        # 연결 풀 최적화
        connector = aiohttp.TCPConnector(
            limit=200,  # 전체 연결 풀 크기
            limit_per_host=100,  # 호스트당 연결 수
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=600,  # 10분 타임아웃
            connect=30,
            sock_read=120
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def upload_project(
        self,
        project_path: str,
        project_id: str = "default",
        project_name: str = None
    ) -> Dict[str, Any]:
        """프로젝트 업로드 및 인덱싱"""
        start_time = time.time()
        
        logger.info(f"🚀 고성능 프로젝트 업로드 시작: {project_path}")
        
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            return {
                "success": False,
                "error": f"프로젝트 경로가 존재하지 않습니다: {project_path}"
            }
        
        if not project_name:
            project_name = project_path.name
        
        # 1. 파일 스캔 (Thread Pool 사용)
        logger.info("📂 파일 스캔 중...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            file_paths = await loop.run_in_executor(
                executor, self._scan_files, project_path
            )
        
        logger.info(f"📋 {len(file_paths)}개 파일 발견")
        
        if not file_paths:
            return {
                "success": False,
                "error": "업로드할 파일이 없습니다."
            }
        
        # 2. 병렬 파일 읽기
        logger.info("📖 파일 읽기 시작...")
        files_data = await self._read_files_parallel(file_paths, project_path)
        
        logger.info(f"✅ {len(files_data)}개 파일 읽기 완료")
        
        if not files_data:
            return {
                "success": False,
                "error": "읽을 수 있는 파일이 없습니다."
            }
        
        # 3. 배치 업로드
        logger.info("📤 배치 업로드 시작...")
        result = await self._upload_batches(files_data, project_id, project_name)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        result["total_time"] = round(total_time, 2)
        result["files_per_second"] = round(len(files_data) / total_time, 1) if total_time > 0 else 0
        
        logger.info(f"🎉 업로드 완료! ({total_time:.2f}초)")
        logger.info(f"   📊 처리 속도: {result['files_per_second']:.1f} 파일/초")
        
        return result
    
    def _scan_files(self, project_path: Path) -> List[Path]:
        """파일 스캔 (동기 함수)"""
        file_paths = []
        
        for root, dirs, files in os.walk(project_path):
            # 무시할 디렉토리 제거
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRECTORIES]
            
            for file in files:
                if file in IGNORE_FILES or file.startswith('.'):
                    continue
                
                file_path = Path(root) / file
                
                # 지원하는 확장자만 처리
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # 파일 크기 체크
                    try:
                        if file_path.stat().st_size <= self.max_file_size:
                            file_paths.append(file_path)
                    except OSError:
                        continue
        
        return file_paths
    
    async def _read_files_parallel(
        self, 
        file_paths: List[Path], 
        project_path: Path
    ) -> List[Dict[str, Any]]:
        """병렬 파일 읽기"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def read_file_with_semaphore(file_path: Path):
            async with semaphore:
                return await self._read_file(file_path, project_path)
        
        # 모든 파일을 병렬로 읽기
        results = await asyncio.gather(
            *[read_file_with_semaphore(fp) for fp in file_paths],
            return_exceptions=True
        )
        
        # 유효한 결과만 필터링
        files_data = []
        for result in results:
            if isinstance(result, dict) and result is not None:
                files_data.append(result)
        
        return files_data
    
    async def _read_file(self, file_path: Path, project_path: Path) -> Dict[str, Any]:
        """단일 파일 읽기"""
        try:
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
            logger.warning(f"파일 읽기 실패 {file_path}: {e}")
            return None
    
    async def _upload_batches(
        self,
        files_data: List[Dict[str, Any]],
        project_id: str,
        project_name: str
    ) -> Dict[str, Any]:
        """배치 업로드"""
        total_batches = math.ceil(len(files_data) / self.batch_size)
        logger.info(f"📦 {total_batches}개 배치로 나누어 업로드...")
        
        # 배치 업로드 태스크 생성
        upload_tasks = []
        for i in range(0, len(files_data), self.batch_size):
            batch = files_data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            task = self._upload_single_batch(batch, batch_num, total_batches, project_id, project_name)
            upload_tasks.append(task)
        
        # 모든 배치를 병렬로 업로드
        batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # 결과 집계
        return self._aggregate_results(batch_results, files_data)
    
    async def _upload_single_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int,
        project_id: str,
        project_name: str
    ) -> Dict[str, Any]:
        """단일 배치 업로드"""
        upload_data = {
            "project_id": project_id,
            "project_name": project_name,
            "files": batch
        }
        
        logger.info(f"📤 배치 {batch_num}/{total_batches} 업로드... ({len(batch)}개 파일)")
        
        upload_url = f"{self.server_url}/api/v1/upload-batch"
        
        try:
            async with self.session.post(upload_url, json=upload_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ 배치 {batch_num} 완료 (성공률: {result.get('success_rate', 0)}%)")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ 배치 {batch_num} 실패: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ 배치 {batch_num} 업로드 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _aggregate_results(
        self, 
        batch_results: List[Any], 
        files_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """결과 집계"""
        total_received = 0
        total_indexed = 0
        total_failed = 0
        failed_batches = 0
        tech_stacks = set()
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"❌ 배치 {i+1} 실패: {result}")
                failed_batches += 1
                continue
            
            if not result.get("success", False):
                failed_batches += 1
                continue
            
            total_received += result.get('total_files_received', 0)
            total_indexed += result.get('indexed_files_count', 0)
            total_failed += result.get('failed_files_count', 0)
            tech_stacks.update(result.get('tech_stack', []))
        
        success_rate = (total_indexed / total_received * 100) if total_received > 0 else 0
        
        return {
            "success": True,
            "total_files_scanned": len(files_data),
            "total_files_received": total_received,
            "indexed_files_count": total_indexed,
            "failed_files_count": total_failed,
            "success_rate": round(success_rate, 1),
            "total_batches": len(batch_results),
            "failed_batches": failed_batches,
            "batch_success_rate": round((len(batch_results) - failed_batches) / len(batch_results) * 100, 1) if batch_results else 0,
            "tech_stack": sorted(list(tech_stacks))
        }

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고성능 프로젝트 업로드")
    parser.add_argument("project_path", help="업로드할 프로젝트 경로")
    parser.add_argument("--project-id", default="default", help="프로젝트 ID")
    parser.add_argument("--project-name", help="프로젝트 이름")
    parser.add_argument("--server-url", default="http://localhost:8000", help="서버 URL")
    
    args = parser.parse_args()
    
    async with FastProjectUploader(args.server_url) as uploader:
        result = await uploader.upload_project(
            args.project_path,
            args.project_id,
            args.project_name
        )
        
        if result["success"]:
            print(f"\n🎉 업로드 성공!")
            print(f"   📊 총 파일: {result['total_files_scanned']}개")
            print(f"   ✅ 인덱싱: {result['indexed_files_count']}개")
            print(f"   📈 성공률: {result['success_rate']:.1f}%")
            print(f"   ⏱️  소요시간: {result['total_time']:.2f}초")
            print(f"   🚀 처리속도: {result['files_per_second']:.1f} 파일/초")
            print(f"   🔧 기술스택: {', '.join(result['tech_stack'])}")
        else:
            print(f"\n❌ 업로드 실패: {result.get('error', '알 수 없는 오류')}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 