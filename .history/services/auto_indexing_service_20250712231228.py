"""
자동 백그라운드 인덱싱 서비스
도커 시작시 자동으로 프로젝트 소스들을 벡터DB에 저장하고 인덱싱
중복 체크 및 증분 업데이트 지원
"""

import os
import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time

from services.vector_service import VectorService
from services.file_indexing_service import FileIndexingService
from services.project_selector_service import ProjectSelectorService
from services.error_handler import handle_errors, ErrorCategory, ErrorLevel
from config import settings

logger = logging.getLogger(__name__)

class AutoIndexingService:
    """자동 백그라운드 인덱싱 서비스"""
    
    def __init__(self, vector_service: VectorService, file_indexing_service: FileIndexingService):
        self.vector_service = vector_service
        self.file_indexing_service = file_indexing_service
        
        # 프로젝트 선택 서비스 초기화
        self.project_selector = ProjectSelectorService(settings.project_whitelist_file)
        
        # 상태 관리
        self.is_running = False
        self.is_indexing = False
        self.current_task = None
        
        # 설정 (config.py에서 가져오기)
        self.host_projects_path = settings.host_projects_path
        self.index_db_path = "/data/index_metadata.json"
        self.scan_interval = settings.auto_indexing_interval
        self.max_workers = settings.max_workers
        self.batch_size = settings.batch_size
        
        # 선택적 인덱싱 설정
        self.selective_indexing_enabled = settings.selective_indexing_enabled
        self.auto_indexing_enabled = settings.auto_indexing_enabled
        
        # 인덱싱 메타데이터 캐시
        self.indexed_files_cache: Dict[str, Dict[str, Any]] = {}
        
        # 지원하는 파일 확장자 (기본값)
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.sass',
            '.json', '.yaml', '.yml', '.md', '.txt', '.sql', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.vue', '.svelte', '.dart', '.r', '.scala', '.clj', '.hs', '.elm',
            '.xml', '.toml', '.ini', '.cfg', '.conf', '.dockerfile',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd'
        }
        
        # 무시할 디렉토리 (기본값)
        self.ignore_directories = {
            '.git', '.svn', '.hg', '.bzr',
            '__pycache__', '.pytest_cache', '.coverage', '.mypy_cache', '.tox',
            'node_modules', '.npm', '.yarn', '.pnpm',
            '.venv', 'venv', '.env', 'env', '.virtualenv',
            'build', 'dist', 'out', 'target', 'bin', 'obj',
            '.idea', '.vscode', '.eclipse', '.vs',
            'logs', 'log', 'tmp', 'temp', '.tmp',
            '.DS_Store', 'Thumbs.db',
            '.next', '.nuxt', '.cache', '.parcel-cache',
            'vendor', 'bower_components',
            'coverage', 'nyc_output',
            'Debug', 'Release',
            '.history',  # Cursor 히스토리
            '.vs',       # Visual Studio
            '.sass-cache',
            '.gradle',
            '.maven'
        }
        
        # 무시할 파일 패턴 (기본값)
        self.ignore_file_patterns = {
            # 로그 파일
            '*.log', '*.logs',
            # 백업 파일
            '*.bak', '*.backup', '*.old', '*.orig',
            # 임시 파일
            '*.tmp', '*.temp', '*~', '*.swp', '*.swo',
            # 바이너리 파일
            '*.exe', '*.dll', '*.so', '*.dylib', '*.bin',
            # 압축 파일
            '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
            # 이미지 파일
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.svg', '*.ico',
            # 미디어 파일
            '*.mp3', '*.mp4', '*.avi', '*.mov', '*.wav',
            # 문서 파일
            '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx', '*.ppt', '*.pptx',
            # 기타
            '*.lock', '*.pid', '.DS_Store', 'Thumbs.db'
        }
        
        logger.info(f"자동 인덱싱 서비스 초기화 완료 (선택적 인덱싱: {'활성화' if self.selective_indexing_enabled else '비활성화'})")
    
    async def start(self):
        """자동 인덱싱 서비스 시작"""
        if self.is_running:
            logger.info("자동 인덱싱 서비스가 이미 실행 중입니다")
            return
        
        if not self.auto_indexing_enabled:
            logger.info("자동 인덱싱이 비활성화되어 있습니다")
            return
        
        self.is_running = True
        logger.info("자동 백그라운드 인덱싱 서비스 시작")
        
        # 프로젝트 선택 서비스 초기화
        await self.project_selector.load_whitelist()
        
        # 초기 인덱싱 메타데이터 로드
        await self._load_index_metadata()
        
        # 초기 스캔 실행
        await self._perform_initial_scan()
        
        # 백그라운드 태스크 시작
        self.current_task = asyncio.create_task(self._background_scan_loop())
        
        logger.info("자동 인덱싱 서비스 시작 완료")
    
    async def stop(self):
        """자동 인덱싱 서비스 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("자동 인덱싱 서비스 중지 중...")
        
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        
        # 메타데이터 저장
        await self._save_index_metadata()
        
        logger.info("자동 인덱싱 서비스 중지 완료")
    
    async def _background_scan_loop(self):
        """백그라운드 스캔 루프"""
        while self.is_running:
            try:
                await asyncio.sleep(self.scan_interval)
                if self.is_running and not self.is_indexing:
                    await self._perform_incremental_scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백그라운드 스캔 중 오류: {e}")
                await asyncio.sleep(60)  # 오류 발생시 1분 대기
    
    async def _perform_initial_scan(self):
        """초기 전체 스캔 수행"""
        if not os.path.exists(self.host_projects_path):
            logger.warning(f"프로젝트 경로가 존재하지 않습니다: {self.host_projects_path}")
            return
        
        logger.info("초기 프로젝트 스캔 시작...")
        start_time = time.time()
        
        try:
            self.is_indexing = True
            
            # 선택적 인덱싱이 활성화된 경우
            if self.selective_indexing_enabled:
                selected_projects = self.project_selector.get_selected_projects()
                if not selected_projects:
                    logger.info("선택된 프로젝트가 없습니다. 인덱싱을 건너뜁니다.")
                    return
                
                logger.info(f"선택된 프로젝트 인덱싱: {selected_projects}")
                project_dirs = []
                
                for project_name in selected_projects:
                    project_path = Path(self.host_projects_path) / project_name
                    if project_path.exists() and project_path.is_dir():
                        project_dirs.append(project_path)
                    else:
                        logger.warning(f"선택된 프로젝트를 찾을 수 없습니다: {project_name}")
            else:
                # 전체 프로젝트 탐지 (기존 방식)
                project_dirs = await self._discover_project_directories()
            
            logger.info(f"인덱싱할 프로젝트 디렉토리: {len(project_dirs)}개")
            
            total_indexed = 0
            total_skipped = 0
            
            for project_dir in project_dirs:
                project_name = project_dir.name
                
                # 프로젝트별 설정 확인
                if self.selective_indexing_enabled:
                    project_config = self.project_selector.get_project_config(project_name)
                    if not project_config or not project_config.get('enabled', True):
                        logger.info(f"프로젝트 '{project_name}' 비활성화됨, 건너뜀")
                        continue
                
                logger.info(f"프로젝트 인덱싱 시작: {project_name}")
                
                indexed_count, skipped_count = await self._index_project_directory(
                    project_dir, project_name
                )
                
                total_indexed += indexed_count
                total_skipped += skipped_count
                
                logger.info(f"프로젝트 '{project_name}' 인덱싱 완료: {indexed_count}개 저장, {skipped_count}개 스킵")
            
            duration = time.time() - start_time
            logger.info(f"초기 스캔 완료: {total_indexed}개 파일 인덱싱, {total_skipped}개 스킵, {duration:.2f}초 소요")
            
        except Exception as e:
            logger.error(f"초기 스캔 중 오류: {e}")
        finally:
            self.is_indexing = False
            await self._save_index_metadata()
    
    async def _perform_incremental_scan(self):
        """증분 스캔 수행 (변경된 파일만)"""
        if not os.path.exists(self.host_projects_path):
            return
        
        logger.info("증분 스캔 시작...")
        start_time = time.time()
        
        try:
            self.is_indexing = True
            
            # 선택적 인덱싱이 활성화된 경우
            if self.selective_indexing_enabled:
                selected_projects = self.project_selector.get_selected_projects()
                if not selected_projects:
                    return
                
                project_dirs = []
                for project_name in selected_projects:
                    project_path = Path(self.host_projects_path) / project_name
                    if project_path.exists() and project_path.is_dir():
                        project_dirs.append(project_path)
            else:
                project_dirs = await self._discover_project_directories()
            
            total_updated = 0
            
            for project_dir in project_dirs:
                project_name = project_dir.name
                
                # 프로젝트별 설정 확인
                if self.selective_indexing_enabled:
                    project_config = self.project_selector.get_project_config(project_name)
                    if not project_config or not project_config.get('enabled', True):
                        continue
                    
                    # 자동 인덱싱이 비활성화된 프로젝트 건너뛰기
                    if not project_config.get('auto_indexing', True):
                        continue
                
                updated_count = await self._scan_project_for_changes(project_dir, project_name)
                total_updated += updated_count
            
            duration = time.time() - start_time
            if total_updated > 0:
                logger.info(f"증분 스캔 완료: {total_updated}개 파일 업데이트, {duration:.2f}초 소요")
            
        except Exception as e:
            logger.error(f"증분 스캔 중 오류: {e}")
        finally:
            self.is_indexing = False
            await self._save_index_metadata()
    
    async def _discover_project_directories(self) -> List[Path]:
        """프로젝트 디렉토리 탐지"""
        project_dirs = []
        
        try:
            base_path = Path(self.host_projects_path)
            
            # 직접 하위 디렉토리들을 프로젝트로 간주
            for item in base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # 프로젝트 디렉토리인지 확인 (소스 파일이 있는지)
                    if await self._is_project_directory(item):
                        project_dirs.append(item)
            
        except Exception as e:
            logger.error(f"프로젝트 디렉토리 탐지 중 오류: {e}")
        
        return project_dirs
    
    async def _is_project_directory(self, path: Path) -> bool:
        """디렉토리가 프로젝트 디렉토리인지 확인"""
        try:
            # 소스 파일이 있는지 확인 (최대 깊이 3 레벨)
            for root, dirs, files in os.walk(path):
                # 무시할 디렉토리 제외
                dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
                
                # 깊이 제한
                level = len(Path(root).relative_to(path).parts)
                if level > 3:
                    continue
                
                # 지원하는 확장자의 파일이 있는지 확인
                for file in files:
                    if Path(file).suffix.lower() in self.supported_extensions:
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"프로젝트 디렉토리 확인 중 오류 {path}: {e}")
            return False
    
    async def _index_project_directory(self, project_dir: Path, project_name: str) -> tuple[int, int]:
        """프로젝트 디렉토리 인덱싱"""
        indexed_count = 0
        skipped_count = 0
        
        try:
            # 소스 파일 수집
            source_files = []
            for root, dirs, files in os.walk(project_dir):
                # 무시할 디렉토리 제외
                dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # 파일 필터링
                    if self._should_index_file(file_path, project_name):
                        source_files.append(file_path)
            
            logger.info(f"프로젝트 '{project_name}'에서 {len(source_files)}개 파일 발견")
            
            # 배치 처리
            for i in range(0, len(source_files), self.batch_size):
                batch = source_files[i:i + self.batch_size]
                batch_indexed, batch_skipped = await self._process_file_batch(batch, project_name)
                indexed_count += batch_indexed
                skipped_count += batch_skipped
                
                # 진행 상황 로그
                if len(source_files) > self.batch_size:
                    progress = min(i + self.batch_size, len(source_files))
                    logger.info(f"프로젝트 '{project_name}' 진행률: {progress}/{len(source_files)} ({progress/len(source_files)*100:.1f}%)")
        
        except Exception as e:
            logger.error(f"프로젝트 '{project_name}' 인덱싱 중 오류: {e}")
        
        return indexed_count, skipped_count
    
    async def _scan_project_for_changes(self, project_dir: Path, project_name: str) -> int:
        """프로젝트에서 변경된 파일 스캔"""
        updated_count = 0
        
        try:
            # 변경된 파일 탐지
            changed_files = []
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    if self._should_index_file(file_path, project_name):
                        if await self._is_file_changed(file_path):
                            changed_files.append(file_path)
            
            if changed_files:
                logger.info(f"프로젝트 '{project_name}'에서 {len(changed_files)}개 변경된 파일 발견")
                
                # 변경된 파일들 배치 처리
                for i in range(0, len(changed_files), self.batch_size):
                    batch = changed_files[i:i + self.batch_size]
                    batch_indexed, _ = await self._process_file_batch(batch, project_name)
                    updated_count += batch_indexed
        
        except Exception as e:
            logger.error(f"프로젝트 '{project_name}' 변경 스캔 중 오류: {e}")
        
        return updated_count
    
    async def _process_file_batch(self, file_batch: List[Path], project_name: str) -> tuple[int, int]:
        """파일 배치 처리"""
        indexed_count = 0
        skipped_count = 0
        
        # 세마포어로 동시 처리 제한
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(file_path: Path):
            async with semaphore:
                return await self._process_single_file(file_path, project_name)
        
        # 병렬 처리
        tasks = [process_single_file(file_path) for file_path in file_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"파일 처리 중 오류: {result}")
                skipped_count += 1
            elif result:
                indexed_count += 1
            else:
                skipped_count += 1
        
        return indexed_count, skipped_count
    
    async def _process_single_file(self, file_path: Path, project_name: str) -> bool:
        """단일 파일 처리"""
        try:
            # 파일 해시 계산
            file_hash = await self._calculate_file_hash(file_path)
            file_key = str(file_path)
            
            # 캐시에서 기존 정보 확인
            cached_info = self.indexed_files_cache.get(file_key)
            
            # 해시가 같으면 스킵
            if cached_info and cached_info.get('hash') == file_hash:
                return False
            
            # 파일 내용 읽기
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            except UnicodeDecodeError:
                # UTF-8로 읽을 수 없는 파일은 스킵
                return False
            
            # 벡터 DB에 저장
            success = await self.file_indexing_service.store_file_content(
                filename=file_path.name,
                content=content,
                project_id=project_name,
                metadata={
                    'file_path': str(file_path.relative_to(Path(self.host_projects_path))),
                    'file_size': len(content),
                    'indexed_at': datetime.now().isoformat(),
                    'auto_indexed': True
                }
            )
            
            if success:
                # 캐시 업데이트
                self.indexed_files_cache[file_key] = {
                    'hash': file_hash,
                    'indexed_at': datetime.now().isoformat(),
                    'project_name': project_name
                }
                return True
            
        except Exception as e:
            logger.error(f"파일 '{file_path}' 처리 중 오류: {e}")
        
        return False
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"파일 '{file_path}' 해시 계산 중 오류: {e}")
            return ""
    
    async def _is_file_changed(self, file_path: Path) -> bool:
        """파일이 변경되었는지 확인"""
        try:
            file_key = str(file_path)
            cached_info = self.indexed_files_cache.get(file_key)
            
            if not cached_info:
                return True  # 새로운 파일
            
            current_hash = await self._calculate_file_hash(file_path)
            return current_hash != cached_info.get('hash')
            
        except Exception as e:
            logger.error(f"파일 변경 확인 중 오류 {file_path}: {e}")
            return True
    
    def _should_index_file(self, file_path: Path, project_name: str = None) -> bool:
        """파일을 인덱싱해야 하는지 확인 (프로젝트별 설정 고려)"""
        try:
            # 프로젝트별 설정 확인
            if self.selective_indexing_enabled and project_name:
                project_config = self.project_selector.get_project_config(project_name)
                if project_config:
                    # 프로젝트별 확장자 설정
                    include_extensions = project_config.get('include_extensions', [])
                    if include_extensions:
                        supported_extensions = set(include_extensions)
                    else:
                        supported_extensions = self.supported_extensions
                    
                    # 프로젝트별 제외 디렉토리 설정
                    exclude_directories = project_config.get('exclude_directories', [])
                    if exclude_directories:
                        ignore_directories = set(exclude_directories)
                    else:
                        ignore_directories = self.ignore_directories
                    
                    # 프로젝트별 파일 크기 제한
                    max_file_size = project_config.get('max_file_size', settings.max_file_size)
                else:
                    supported_extensions = self.supported_extensions
                    ignore_directories = self.ignore_directories
                    max_file_size = settings.max_file_size
            else:
                supported_extensions = self.supported_extensions
                ignore_directories = self.ignore_directories
                max_file_size = settings.max_file_size
            
            # 확장자 확인
            if file_path.suffix.lower() not in supported_extensions:
                return False
            
            # 파일명 패턴 확인
            file_name = file_path.name.lower()
            for pattern in self.ignore_file_patterns:
                if file_name.endswith(pattern.replace('*', '')):
                    return False
            
            # 숨김 파일 확인
            if file_name.startswith('.'):
                return False
            
            # 제외 디렉토리 확인
            for part in file_path.parts:
                if part in ignore_directories:
                    return False
            
            # 파일 크기 확인
            try:
                if file_path.stat().st_size > max_file_size:
                    return False
            except:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"파일 인덱싱 여부 확인 중 오류 {file_path}: {e}")
            return False
    
    async def _load_index_metadata(self):
        """인덱싱 메타데이터 로드"""
        try:
            if os.path.exists(self.index_db_path):
                async with aiofiles.open(self.index_db_path, 'r') as f:
                    data = await f.read()
                    self.indexed_files_cache = json.loads(data)
                logger.info(f"인덱싱 메타데이터 로드 완료: {len(self.indexed_files_cache)}개 파일")
            else:
                self.indexed_files_cache = {}
                logger.info("새로운 인덱싱 메타데이터 시작")
        except Exception as e:
            logger.error(f"인덱싱 메타데이터 로드 중 오류: {e}")
            self.indexed_files_cache = {}
    
    async def _save_index_metadata(self):
        """인덱싱 메타데이터 저장"""
        try:
            # 데이터 디렉토리 생성
            os.makedirs(os.path.dirname(self.index_db_path), exist_ok=True)
            
            async with aiofiles.open(self.index_db_path, 'w') as f:
                await f.write(json.dumps(self.indexed_files_cache, indent=2))
            
            logger.info(f"인덱싱 메타데이터 저장 완료: {len(self.indexed_files_cache)}개 파일")
        except Exception as e:
            logger.error(f"인덱싱 메타데이터 저장 중 오류: {e}")
    
    async def add_project_to_whitelist(self, project_name: str, config: Dict[str, Any] = None) -> bool:
        """프로젝트를 화이트리스트에 추가"""
        return await self.project_selector.add_project(project_name, config)
    
    async def remove_project_from_whitelist(self, project_name: str) -> bool:
        """프로젝트를 화이트리스트에서 제거"""
        return await self.project_selector.remove_project(project_name)
    
    async def get_available_projects(self) -> List[Dict[str, Any]]:
        """사용 가능한 프로젝트 목록 조회"""
        return await self.project_selector.discover_available_projects(self.host_projects_path)
    
    async def get_selected_projects(self) -> List[str]:
        """선택된 프로젝트 목록 조회"""
        return self.project_selector.get_selected_projects()
    
    async def update_project_config(self, project_name: str, config: Dict[str, Any]) -> bool:
        """프로젝트 설정 업데이트"""
        return await self.project_selector.update_project_config(project_name, config)
    
    async def force_index_project(self, project_name: str) -> Dict[str, Any]:
        """특정 프로젝트 강제 인덱싱"""
        project_path = Path(self.host_projects_path) / project_name
        if not project_path.exists():
            return {
                "success": False,
                "error": f"프로젝트를 찾을 수 없습니다: {project_name}"
            }
        
        logger.info(f"강제 인덱싱 시작: {project_name}")
        
        try:
            indexed_count, skipped_count = await self._index_project_directory(project_path, project_name)
            
            return {
                "success": True,
                "project_name": project_name,
                "indexed_files": indexed_count,
                "skipped_files": skipped_count
            }
        except Exception as e:
            logger.error(f"강제 인덱싱 실패 {project_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """자동 인덱싱 서비스 상태 조회"""
        project_selector_status = await self.project_selector.get_status()
        
        return {
            "is_running": self.is_running,
            "is_indexing": self.is_indexing,
            "auto_indexing_enabled": self.auto_indexing_enabled,
            "selective_indexing_enabled": self.selective_indexing_enabled,
            "indexed_files_count": len(self.indexed_files_cache),
            "scan_interval": self.scan_interval,
            "supported_extensions": list(self.supported_extensions),
            "host_projects_path": self.host_projects_path,
            "last_scan_time": max(
                (info.get('indexed_at', '') for info in self.indexed_files_cache.values()),
                default='없음'
            ),
            **project_selector_status
        } 