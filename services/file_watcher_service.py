"""
파일 변경 감지 및 자동 업로드 서비스
사용자가 새로운 코드를 생성하거나 수정할 때 자동으로 MCP 서버로 전송
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import aiofiles
import hashlib
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import time
from services.vector_service import VectorService
from services.error_handler import handle_errors, ErrorCategory, ErrorLevel
from config import settings

logger = logging.getLogger(__name__)

class FileWatcherService:
    """파일 변경 감지 및 자동 업로드 서비스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.observers: Dict[str, Observer] = {}
        self.watched_projects: Dict[str, Dict[str, Any]] = {}
        self.file_hashes: Dict[str, str] = {}
        self.debounce_time = 2.0  # 2초 디바운스
        self.last_processed: Dict[str, float] = {}
        
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
            # 일반적인 무시 패턴
            '.git', '.svn', '.hg', '.bzr',
            '__pycache__', '.pytest_cache', '.coverage',
            'node_modules', '.npm', '.yarn',
            '.venv', 'venv', '.env', 'env',
            'build', 'dist', 'out', 'target',
            '.idea', '.vscode', '.eclipse',
            'logs', 'log', 'tmp', 'temp',
            '.DS_Store', 'Thumbs.db',
            # 추가 무시 패턴
            '.next', '.nuxt', '.cache',
            'vendor', 'bower_components',
            'coverage', 'nyc_output',
            '.mypy_cache', '.tox',
            'bin', 'obj', 'Debug', 'Release'
        }
        
        logger.info("파일 워처 서비스 초기화 완료")
    
    class ProjectFileHandler(FileSystemEventHandler):
        """프로젝트 파일 이벤트 핸들러"""
        
        def __init__(self, watcher_service: 'FileWatcherService', project_id: str):
            super().__init__()
            self.watcher_service = watcher_service
            self.project_id = project_id
        
        def on_modified(self, event):
            if not event.is_directory:
                asyncio.create_task(
                    self.watcher_service._handle_file_event(event.src_path, 'modified', self.project_id)
                )
        
        def on_created(self, event):
            if not event.is_directory:
                asyncio.create_task(
                    self.watcher_service._handle_file_event(event.src_path, 'created', self.project_id)
                )
        
        def on_deleted(self, event):
            if not event.is_directory:
                asyncio.create_task(
                    self.watcher_service._handle_file_event(event.src_path, 'deleted', self.project_id)
                )
    
    @handle_errors(
        category=ErrorCategory.SYSTEM,
        level=ErrorLevel.MEDIUM,
        user_message="프로젝트 감시 시작 중 오류가 발생했습니다."
    )
    async def start_watching_project(
        self,
        project_path: str,
        project_id: str = "default",
        recursive: bool = True,
        auto_upload: bool = True
    ) -> Dict[str, Any]:
        """
        프로젝트 디렉토리 감시 시작
        
        Args:
            project_path: 감시할 프로젝트 경로
            project_id: 프로젝트 ID
            recursive: 하위 디렉토리 포함 여부
            auto_upload: 자동 업로드 여부
            
        Returns:
            감시 시작 결과
        """
        try:
            # 경로 존재 확인
            if not os.path.exists(project_path):
                return {
                    "success": False,
                    "error": f"프로젝트 경로가 존재하지 않습니다: {project_path}",
                    "project_id": project_id
                }
            
            # 이미 감시 중인지 확인
            if project_id in self.observers:
                logger.info(f"프로젝트 {project_id}는 이미 감시 중입니다")
                return {
                    "success": True,
                    "message": "이미 감시 중인 프로젝트입니다",
                    "project_id": project_id,
                    "project_path": project_path
                }
            
            # 파일 핸들러 생성
            event_handler = self.ProjectFileHandler(self, project_id)
            
            # 옵저버 생성 및 시작
            observer = Observer()
            observer.schedule(event_handler, project_path, recursive=recursive)
            observer.start()
            
            # 프로젝트 정보 저장
            self.observers[project_id] = observer
            self.watched_projects[project_id] = {
                "path": project_path,
                "recursive": recursive,
                "auto_upload": auto_upload,
                "started_at": datetime.now().isoformat(),
                "file_count": 0,
                "upload_count": 0,
                "last_activity": None
            }
            
            # 기존 파일들의 해시 계산
            await self._calculate_initial_hashes(project_path, project_id)
            
            logger.info(f"프로젝트 감시 시작: {project_id} -> {project_path}")
            
            return {
                "success": True,
                "message": "프로젝트 감시를 시작했습니다",
                "project_id": project_id,
                "project_path": project_path,
                "recursive": recursive,
                "auto_upload": auto_upload,
                "started_at": self.watched_projects[project_id]["started_at"]
            }
            
        except Exception as e:
            logger.error(f"프로젝트 감시 시작 중 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id,
                "project_path": project_path
            }
    
    @handle_errors(
        category=ErrorCategory.SYSTEM,
        level=ErrorLevel.MEDIUM,
        user_message="프로젝트 감시 중지 중 오류가 발생했습니다."
    )
    async def stop_watching_project(self, project_id: str) -> Dict[str, Any]:
        """
        프로젝트 감시 중지
        
        Args:
            project_id: 프로젝트 ID
            
        Returns:
            감시 중지 결과
        """
        try:
            if project_id not in self.observers:
                return {
                    "success": False,
                    "error": f"감시 중이지 않은 프로젝트입니다: {project_id}",
                    "project_id": project_id
                }
            
            # 옵저버 중지
            observer = self.observers[project_id]
            observer.stop()
            observer.join()
            
            # 프로젝트 정보 정리
            project_info = self.watched_projects.pop(project_id, {})
            del self.observers[project_id]
            
            logger.info(f"프로젝트 감시 중지: {project_id}")
            
            return {
                "success": True,
                "message": "프로젝트 감시를 중지했습니다",
                "project_id": project_id,
                "project_info": project_info
            }
            
        except Exception as e:
            logger.error(f"프로젝트 감시 중지 중 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }
    
    async def _handle_file_event(self, file_path: str, event_type: str, project_id: str):
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
                current_hash = await self._calculate_file_hash(file_path)
                if file_path in self.file_hashes:
                    if self.file_hashes[file_path] == current_hash:
                        return  # 내용이 변경되지 않음
                
                self.file_hashes[file_path] = current_hash
                
                # 자동 업로드 처리
                if self.watched_projects[project_id]["auto_upload"]:
                    await self._auto_upload_file(file_path, project_id, event_type)
            
            elif event_type == 'deleted':
                # 삭제된 파일 해시 제거
                if file_path in self.file_hashes:
                    del self.file_hashes[file_path]
                
                # 벡터 DB에서 제거
                await self._remove_file_from_vector_db(file_path, project_id)
            
            # 프로젝트 정보 업데이트
            self.watched_projects[project_id]["last_activity"] = datetime.now().isoformat()
            self.watched_projects[project_id]["file_count"] += 1
            
            logger.info(f"파일 이벤트 처리: {event_type} -> {file_path}")
            
        except Exception as e:
            logger.error(f"파일 이벤트 처리 중 오류: {str(e)}")
    
    async def _auto_upload_file(self, file_path: str, project_id: str, event_type: str):
        """파일 자동 업로드"""
        try:
            # 파일 읽기
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # 메타데이터 구성
            metadata = {
                'source': os.path.basename(file_path),
                'file_path': file_path,
                'type': 'auto_uploaded_file',
                'project_id': project_id,
                'event_type': event_type,
                'upload_timestamp': datetime.now().isoformat(),
                'file_size': len(content),
                'file_extension': Path(file_path).suffix.lower()
            }
            
            # 벡터 DB에 저장
            success = await self.vector_service.store_document(
                content=content,
                metadata=metadata,
                project_id=project_id
            )
            
            if success:
                self.watched_projects[project_id]["upload_count"] += 1
                logger.info(f"파일 자동 업로드 완료: {file_path}")
            else:
                logger.warning(f"파일 자동 업로드 실패: {file_path}")
                
        except Exception as e:
            logger.error(f"파일 자동 업로드 중 오류: {str(e)}")
    
    async def _remove_file_from_vector_db(self, file_path: str, project_id: str):
        """벡터 DB에서 파일 제거"""
        try:
            # 파일 경로로 문서 검색 후 제거
            # 이는 vector_service에 구현되어야 하는 기능
            logger.info(f"벡터 DB에서 파일 제거: {file_path}")
            
        except Exception as e:
            logger.error(f"벡터 DB에서 파일 제거 중 오류: {str(e)}")
    
    async def _calculate_initial_hashes(self, project_path: str, project_id: str):
        """초기 파일 해시 계산"""
        try:
            for root, dirs, files in os.walk(project_path):
                # 무시할 디렉토리 제거
                dirs[:] = [d for d in dirs if not self._should_ignore_dir(d)]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if not self._should_ignore_file(file_path):
                        file_hash = await self._calculate_file_hash(file_path)
                        self.file_hashes[file_path] = file_hash
                        
        except Exception as e:
            logger.error(f"초기 파일 해시 계산 중 오류: {str(e)}")
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 중 오류: {str(e)}")
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
    
    def _should_ignore_dir(self, dir_name: str) -> bool:
        """디렉토리가 무시 대상인지 확인"""
        return dir_name in self.ignore_patterns or dir_name.startswith('.')
    
    async def get_watching_status(self) -> Dict[str, Any]:
        """감시 상태 조회"""
        try:
            status = {
                "total_projects": len(self.watched_projects),
                "active_observers": len(self.observers),
                "projects": {}
            }
            
            for project_id, project_info in self.watched_projects.items():
                status["projects"][project_id] = {
                    **project_info,
                    "is_active": project_id in self.observers
                }
            
            return {
                "success": True,
                "status": status
            }
            
        except Exception as e:
            logger.error(f"감시 상태 조회 중 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop_all_watchers(self):
        """모든 감시자 중지"""
        try:
            for project_id in list(self.observers.keys()):
                await self.stop_watching_project(project_id)
            
            logger.info("모든 파일 감시자 중지 완료")
            
        except Exception as e:
            logger.error(f"모든 감시자 중지 중 오류: {str(e)}")
    
    def __del__(self):
        """소멸자 - 모든 감시자 정리"""
        try:
            # 동기적으로 모든 감시자 중지
            for project_id, observer in self.observers.items():
                if observer and observer.is_alive():
                    observer.stop()
                    observer.join(timeout=1.0)  # 최대 1초 대기
                    logger.debug(f"프로젝트 {project_id} 감시자 정리 완료")
            
            self.observers.clear()
            self.watched_projects.clear()
            self.file_hashes.clear()
            self.last_processed.clear()
            
        except Exception as e:
            logger.warning(f"감시자 정리 중 오류: {e}") 