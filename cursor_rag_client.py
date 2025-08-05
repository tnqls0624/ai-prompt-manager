#!/usr/bin/env python3
"""
Cursor RAG 클라이언트
Cursor 에디터와 MCP 서버를 연동하여 지능형 프롬프트 시스템을 제공
"""

import asyncio
import aiohttp
import os
import json
import sys
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime
import time
import hashlib
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalFileWatcher:
    """로컬 파일 시스템 감시자 (호스트에서 실행)"""
    
    def __init__(self, mcp_client: 'CursorRAGClient', project_path: str, project_id: str):
        self.mcp_client = mcp_client
        self.project_path = project_path
        self.project_id = project_id
        self.observer = None
        self.file_hashes = {}
        self.debounce_time = 2.0
        self.last_processed = {}
        
        # 감지할 파일 확장자
        self.watched_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.sass',
            '.json', '.yaml', '.yml', '.md', '.txt', '.sql', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.vue', '.svelte', '.dart', '.r', '.scala', '.clj', '.hs', '.elm',
            '.xml', '.toml', '.ini', '.cfg', '.conf', '.env', '.dockerfile',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd'
        }
        
        # 무시할 디렉토리 및 파일
        self.ignore_patterns = {
            '.git', '.svn', '.hg', '.bzr',
            '__pycache__', '.pytest_cache', '.coverage',
            'node_modules', '.npm', '.yarn',
            '.venv', 'venv', '.env', 'env',
            'build', 'dist', 'out', 'target',
            '.idea', '.vscode', '.eclipse',
            'logs', 'log', 'tmp', 'temp',
            '.DS_Store', 'Thumbs.db',
            '.next', '.nuxt', '.cache',
            'vendor', 'bower_components',
            'coverage', 'nyc_output',
            '.mypy_cache', '.tox',
            'bin', 'obj', 'Debug', 'Release'
        }
    
    class FileEventHandler(FileSystemEventHandler):
        """파일 이벤트 핸들러"""
        
        def __init__(self, watcher: 'LocalFileWatcher'):
            super().__init__()
            self.watcher = watcher
        
        def on_modified(self, event):
            if not event.is_directory:
                asyncio.create_task(self.watcher._handle_file_event(event.src_path, 'modified'))
        
        def on_created(self, event):
            if not event.is_directory:
                asyncio.create_task(self.watcher._handle_file_event(event.src_path, 'created'))
        
        def on_deleted(self, event):
            if not event.is_directory:
                asyncio.create_task(self.watcher._handle_file_event(event.src_path, 'deleted'))
    
    def start_watching(self):
        """파일 감시 시작"""
        try:
            if not os.path.exists(self.project_path):
                logger.error(f"❌ 프로젝트 경로가 존재하지 않습니다: {self.project_path}")
                return False
            
            self.observer = Observer()
            event_handler = self.FileEventHandler(self)
            self.observer.schedule(event_handler, self.project_path, recursive=True)
            self.observer.start()
            
            logger.info(f"✅ 파일 감시 시작: {self.project_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 파일 감시 시작 실패: {str(e)}")
            return False
    
    def stop_watching(self):
        """파일 감시 중지"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
                logger.info("✅ 파일 감시 중지")
                return True
        except Exception as e:
            logger.error(f"❌ 파일 감시 중지 실패: {str(e)}")
        return False
    
    async def _handle_file_event(self, file_path: str, event_type: str):
        """파일 이벤트 처리"""
        try:
            # 파일 확장자 확인
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.watched_extensions:
                return
            
            # 무시 패턴 확인
            if self._should_ignore_file(file_path):
                return
            
            # 디바운스 처리
            current_time = time.time()
            if file_path in self.last_processed:
                if current_time - self.last_processed[file_path] < self.debounce_time:
                    return
            
            self.last_processed[file_path] = current_time
            
            # 파일 해시 확인 (중복 처리 방지)
            if event_type in ['modified', 'created']:
                if not os.path.exists(file_path):
                    return
                    
                current_hash = await self._calculate_file_hash(file_path)
                if file_path in self.file_hashes:
                    if self.file_hashes[file_path] == current_hash:
                        return  # 내용이 변경되지 않음
                
                self.file_hashes[file_path] = current_hash
                
                # 파일 업로드
                await self._upload_file(file_path, event_type)
            
            elif event_type == 'deleted':
                # 삭제된 파일 해시 제거
                if file_path in self.file_hashes:
                    del self.file_hashes[file_path]
            
            logger.info(f"📁 파일 이벤트 처리: {event_type} -> {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"❌ 파일 이벤트 처리 중 오류: {str(e)}")
    
    async def _upload_file(self, file_path: str, event_type: str):
        """파일을 MCP 서버로 업로드"""
        try:
            # 파일 읽기
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # 파일 정보 구성
            file_info = {
                'filename': os.path.basename(file_path),
                'content': content,
                'metadata': {
                    'file_path': file_path,
                    'event_type': event_type,
                    'project_id': self.project_id,
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_size': len(content),
                    'file_extension': Path(file_path).suffix.lower()
                }
            }
            
            # HTTP API로 업로드
            async with self.mcp_client.session.post(
                f"{self.mcp_client.mcp_server_url}/api/v1/upload-files",
                json={
                    'project_id': self.project_id,
                    'files': [file_info]
                }
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ 파일 업로드 성공: {os.path.basename(file_path)}")
                else:
                    logger.error(f"❌ 파일 업로드 실패: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ 파일 업로드 중 오류: {str(e)}")
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"❌ 파일 해시 계산 중 오류: {str(e)}")
            return ""
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """파일이 무시 대상인지 확인"""
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        # 확장자 확인
        if file_ext not in self.watched_extensions:
            return True
        
        # 숨김 파일 확인
        if file_name.startswith('.'):
            return True
        
        # 무시 패턴 확인
        for pattern in self.ignore_patterns:
            if pattern in file_path:
                return True
        
        return False

logger = logging.getLogger(__name__)

class CursorRAGClient:
    """Cursor와 MCP 서버를 연동하는 클라이언트"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.project_id = "default"
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃
        self.file_watcher: Optional[LocalFileWatcher] = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 시작"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        # 파일 와처 정리
        if self.file_watcher:
            self.file_watcher.stop_watching()
            self.file_watcher = None
        
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        try:
            async with self.session.get(f"{self.mcp_server_url}/api/v1/heartbeat") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ MCP 서버 상태: 정상")
                    return {"success": True, "status": result}
                else:
                    logger.error(f"❌ MCP 서버 상태 확인 실패: {response.status}")
                    return {"success": False, "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ MCP 서버 연결 실패: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def start_file_watching(
        self,
        project_path: str,
        project_id: str = "default",
        recursive: bool = True,
        auto_upload: bool = True
    ) -> Dict[str, Any]:
        """파일 감시 시작 (로컬 파일 와처 사용)"""
        try:
            # 이미 감시 중인 경우 중지
            if self.file_watcher:
                self.file_watcher.stop_watching()
                self.file_watcher = None
            
            # 새로운 파일 와처 생성
            self.file_watcher = LocalFileWatcher(self, project_path, project_id)
            
            # 파일 감시 시작
            success = self.file_watcher.start_watching()
            
            if success:
                logger.info(f"✅ 파일 감시 시작: {project_path}")
                print(f"📁 프로젝트 감시 시작")
                print(f"   경로: {project_path}")
                print(f"   프로젝트 ID: {project_id}")
                print(f"   하위 디렉토리 포함: {recursive}")
                print(f"   자동 업로드: {auto_upload}")
                print(f"   💡 파일 변경 시 자동으로 MCP 서버로 업로드됩니다")
                
                return {
                    "success": True,
                    "message": "로컬 파일 감시를 시작했습니다",
                    "project_path": project_path,
                    "project_id": project_id,
                    "recursive": recursive,
                    "auto_upload": auto_upload,
                    "started_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "파일 감시 시작에 실패했습니다",
                    "project_path": project_path,
                    "project_id": project_id
                }
                
        except Exception as e:
            logger.error(f"❌ 파일 감시 시작 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def stop_file_watching(self, project_id: str = "default") -> Dict[str, Any]:
        """파일 감시 중지 (로컬 파일 와처 사용)"""
        try:
            if self.file_watcher:
                success = self.file_watcher.stop_watching()
                self.file_watcher = None
                
                if success:
                    logger.info(f"✅ 파일 감시 중지: {project_id}")
                    print(f"⏹️  프로젝트 감시 중지: {project_id}")
                    return {
                        "success": True,
                        "message": "로컬 파일 감시를 중지했습니다",
                        "project_id": project_id
                    }
                else:
                    return {
                        "success": False,
                        "error": "파일 감시 중지에 실패했습니다",
                        "project_id": project_id
                    }
            else:
                return {
                    "success": False,
                    "error": "실행 중인 파일 감시자가 없습니다",
                    "project_id": project_id
                }
                
        except Exception as e:
            logger.error(f"❌ 파일 감시 중지 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_watcher_status(self) -> Dict[str, Any]:
        """파일 감시 상태 조회 (로컬 파일 와처 사용)"""
        try:
            if self.file_watcher:
                is_active = self.file_watcher.observer is not None
                
                print(f"📊 로컬 파일 감시 상태")
                print(f"   상태: {'활성' if is_active else '비활성'}")
                print(f"   프로젝트 경로: {self.file_watcher.project_path}")
                print(f"   프로젝트 ID: {self.file_watcher.project_id}")
                print(f"   감시 중인 파일 수: {len(self.file_watcher.file_hashes)}")
                print(f"   처리된 이벤트 수: {len(self.file_watcher.last_processed)}")
                
                return {
                    "success": True,
                    "status": {
                        "is_active": is_active,
                        "project_path": self.file_watcher.project_path,
                        "project_id": self.file_watcher.project_id,
                        "watched_files_count": len(self.file_watcher.file_hashes),
                        "processed_events_count": len(self.file_watcher.last_processed),
                        "watched_extensions": list(self.file_watcher.watched_extensions),
                        "ignore_patterns": list(self.file_watcher.ignore_patterns)
                    }
                }
            else:
                print(f"📊 로컬 파일 감시 상태")
                print(f"   상태: 실행 중이지 않음")
                
                return {
                    "success": True,
                    "status": {
                        "is_active": False,
                        "message": "실행 중인 파일 감시자가 없습니다"
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ 감시 상태 조회 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def enhance_prompt_with_rag(
        self,
        prompt: str,
        project_id: str = "default",
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """RAG 기반 프롬프트 개선"""
        try:
            payload = {
                "prompt": prompt,
                "project_id": project_id,
                "context_limit": context_limit
            }
            
            async with self.session.post(
                f"{self.mcp_server_url}/api/v1/rag/enhance-prompt",
                json=payload
            ) as response:
                result = await response.json()
                
                if result.get("success"):
                    logger.info("✅ RAG 기반 프롬프트 개선 완료")
                    print("🎯 프롬프트 개선 완료!")
                    print(f"📄 원본 프롬프트: {result.get('original_prompt', '')[:100]}...")
                    print(f"📊 컨텍스트 개수: {result.get('metadata', {}).get('context_count', 0)}")
                    print(f"🔍 컨텍스트 소스: {', '.join(result.get('metadata', {}).get('context_sources', []))}")
                    print("\n" + "="*80)
                    print("🚀 향상된 프롬프트:")
                    print("="*80)
                    print(result.get("enhanced_prompt", ""))
                    print("="*80)
                else:
                    logger.error(f"❌ 프롬프트 개선 실패: {result.get('error')}")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ 프롬프트 개선 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate_code_with_rag(
        self,
        prompt: str,
        project_id: str = "default",
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """RAG 기반 코드 생성"""
        try:
            payload = {
                "prompt": prompt,
                "project_id": project_id,
                "context_limit": context_limit
            }
            
            async with self.session.post(
                f"{self.mcp_server_url}/api/v1/rag/generate-code",
                json=payload
            ) as response:
                result = await response.json()
                
                if result.get("success"):
                    logger.info("✅ RAG 기반 코드 생성 완료")
                    print("🎯 코드 생성 완료!")
                    print(f"📄 요청사항: {result.get('original_prompt', '')[:100]}...")
                    print(f"📊 컨텍스트 개수: {result.get('metadata', {}).get('context_count', 0)}")
                    print(f"🔍 컨텍스트 소스: {', '.join(result.get('metadata', {}).get('context_sources', []))}")
                    print("\n" + "="*80)
                    print("💻 생성된 코드:")
                    print("="*80)
                    print(result.get("generated_code", ""))
                    print("="*80)
                else:
                    logger.error(f"❌ 코드 생성 실패: {result.get('error')}")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ 코드 생성 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def search_and_summarize(
        self,
        query: str,
        project_id: str = "default",
        limit: int = 3
    ) -> Dict[str, Any]:
        """RAG 기반 검색 및 요약"""
        try:
            payload = {
                "query": query,
                "project_id": project_id,
                "limit": limit
            }
            
            async with self.session.post(
                f"{self.mcp_server_url}/api/v1/rag/search-summarize",
                json=payload
            ) as response:
                result = await response.json()
                
                if result.get("success"):
                    logger.info("✅ RAG 기반 검색 및 요약 완료")
                    print("🔍 검색 및 요약 완료!")
                    print(f"📄 쿼리: {result.get('query', '')}")
                    print(f"📊 관련 문서 수: {result.get('document_count', 0)}")
                    print("\n" + "="*80)
                    print("📋 요약 결과:")
                    print("="*80)
                    print(result.get("summary", ""))
                    print("="*80)
                    print("📚 관련 문서:")
                    for doc in result.get("relevant_documents", []):
                        print(f"   📄 {doc.get('source', 'Unknown')}: {doc.get('content', '')[:100]}...")
                else:
                    logger.error(f"❌ 검색 및 요약 실패: {result.get('error')}")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ 검색 및 요약 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def upload_current_project(self, project_path: str, project_id: str = "default") -> Dict[str, Any]:
        """현재 프로젝트 업로드 (기존 upload_project.py 기능)"""
        try:
            # 기존 업로드 스크립트 실행
            import subprocess
            result = subprocess.run([
                sys.executable, "upload_project.py",
                "--project-path", project_path,
                "--project-id", project_id
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ 프로젝트 업로드 완료")
                print("📤 프로젝트 업로드 완료!")
                print(result.stdout)
                return {"success": True, "message": "프로젝트 업로드 완료"}
            else:
                logger.error(f"❌ 프로젝트 업로드 실패: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ 프로젝트 업로드 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}

def create_cursor_config():
    """Cursor 설정 파일 생성"""
    config = {
        "mcp_server_url": "http://localhost:8000",
        "project_id": "default",
        "auto_watch": True,
        "context_limit": 5,
        "ignore_patterns": [
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "build", "dist", "target", ".idea", ".vscode"
        ]
    }
    
    config_path = Path.cwd() / ".cursor-rag-config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Cursor RAG 설정 파일 생성: {config_path}")
    return config_path

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Cursor RAG 클라이언트")
    parser.add_argument("--server-url", default="http://localhost:8000", help="MCP 서버 URL")
    parser.add_argument("--project-id", default="default", help="프로젝트 ID")
    parser.add_argument("--project-path", default=".", help="프로젝트 경로")
    
    # 서브명령어 설정
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 헬스체크
    subparsers.add_parser('health', help='서버 상태 확인')
    
    # 파일 감시 관련
    watch_parser = subparsers.add_parser('watch', help='파일 감시 시작')
    watch_parser.add_argument('--no-auto-upload', action='store_true', help='자동 업로드 비활성화')
    watch_parser.add_argument('--no-recursive', action='store_true', help='하위 디렉토리 감시 비활성화')
    watch_parser.add_argument('--keep-alive', action='store_true', help='파일 감시를 계속 실행 (Ctrl+C로 중지)')
    
    subparsers.add_parser('unwatch', help='파일 감시 중지')
    subparsers.add_parser('status', help='파일 감시 상태 조회')
    
    # 프롬프트 개선
    enhance_parser = subparsers.add_parser('enhance', help='RAG 기반 프롬프트 개선')
    enhance_parser.add_argument('prompt', help='개선할 프롬프트')
    enhance_parser.add_argument('--context-limit', type=int, default=5, help='컨텍스트 개수')
    
    # 코드 생성
    generate_parser = subparsers.add_parser('generate', help='RAG 기반 코드 생성')
    generate_parser.add_argument('prompt', help='코드 생성 요청')
    generate_parser.add_argument('--context-limit', type=int, default=5, help='컨텍스트 개수')
    
    # 검색 및 요약
    search_parser = subparsers.add_parser('search', help='RAG 기반 검색 및 요약')
    search_parser.add_argument('query', help='검색 쿼리')
    search_parser.add_argument('--limit', type=int, default=3, help='검색 결과 개수')
    
    # 프로젝트 업로드
    subparsers.add_parser('upload', help='현재 프로젝트 업로드')
    
    # 설정 파일 생성
    subparsers.add_parser('init', help='Cursor RAG 설정 파일 생성')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        create_cursor_config()
        return
    
    # 클라이언트 실행
    async with CursorRAGClient(args.server_url) as client:
        client.project_id = args.project_id
        
        if args.command == 'health':
            await client.health_check()
            
        elif args.command == 'watch':
            result = await client.start_file_watching(
                project_path=args.project_path,
                project_id=args.project_id,
                recursive=not args.no_recursive,
                auto_upload=not args.no_auto_upload
            )
            
            # --keep-alive 옵션이 있으면 파일 감시를 계속 실행
            if args.keep_alive and result.get("success"):
                print("📡 파일 감시가 실행 중입니다. Ctrl+C로 중지하세요...")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 파일 감시를 중지합니다...")
                    await client.stop_file_watching(args.project_id)
            
        elif args.command == 'unwatch':
            await client.stop_file_watching(args.project_id)
            
        elif args.command == 'status':
            await client.get_watcher_status()
            
        elif args.command == 'enhance':
            await client.enhance_prompt_with_rag(
                prompt=args.prompt,
                project_id=args.project_id,
                context_limit=args.context_limit
            )
            
        elif args.command == 'generate':
            await client.generate_code_with_rag(
                prompt=args.prompt,
                project_id=args.project_id,
                context_limit=args.context_limit
            )
            
        elif args.command == 'search':
            await client.search_and_summarize(
                query=args.query,
                project_id=args.project_id,
                limit=args.limit
            )
            
        elif args.command == 'upload':
            await client.upload_current_project(
                project_path=args.project_path,
                project_id=args.project_id
            )
            
        else:
            parser.print_help()

if __name__ == "__main__":
    print("🚀 Cursor RAG 클라이언트 시작")
    print("="*50)
    asyncio.run(main()) 