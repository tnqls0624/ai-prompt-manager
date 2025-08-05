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
from services.error_handler import handle_errors, ErrorCategory, ErrorLevel
from models.prompt_models import ProjectContext
from config import settings

logger = logging.getLogger(__name__)

class AutoIndexingService:
    """자동 백그라운드 인덱싱 서비스"""
    
    def __init__(self, vector_service: VectorService, file_indexing_service: FileIndexingService):
        self.vector_service = vector_service
        self.file_indexing_service = file_indexing_service
        
        # 상태 관리
        self.is_running = False
        self.is_indexing = False
        self.current_task = None
        
        # 설정
        self.host_projects_path = "/host_projects"  # Docker 볼륨 마운트 경로
        self.index_db_path = "/data/index_metadata.json"  # 인덱싱 메타데이터 저장 경로
        self.scan_interval = 300  # 5분마다 스캔 (설정 가능)
        self.max_workers = 10  # 병렬 처리 워커 수
        self.batch_size = 50   # 배치 처리 크기
        
        # 인덱싱 메타데이터 캐시
        self.indexed_files_cache: Dict[str, Dict[str, Any]] = {}
        
        # 지원하는 파일 확장자
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.sass',
            '.json', '.yaml', '.yml', '.md', '.txt', '.sql', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.vue', '.svelte', '.dart', '.r', '.scala', '.clj', '.hs', '.elm',
            '.xml', '.toml', '.ini', '.cfg', '.conf', '.dockerfile',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd'
        }
        
        # 무시할 디렉토리 (더 엄격한 필터링)
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
        
        # 무시할 파일 패턴
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
        
        logger.info("자동 인덱싱 서비스 초기화 완료")
    
    async def start(self):
        """자동 인덱싱 서비스 시작"""
        if self.is_running:
            logger.info("자동 인덱싱 서비스가 이미 실행 중입니다")
            return
        
        self.is_running = True
        logger.info("자동 백그라운드 인덱싱 서비스 시작")
        
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
            
            # 프로젝트 디렉토리 탐지
            project_dirs = await self._discover_project_directories()
            logger.info(f"발견된 프로젝트 디렉토리: {len(project_dirs)}개")
            
            total_indexed = 0
            total_skipped = 0
            
            for project_dir in project_dirs:
                project_name = project_dir.name
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
            
            project_dirs = await self._discover_project_directories()
            total_updated = 0
            
            for project_dir in project_dirs:
                project_name = project_dir.name
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
            # 프로젝트 컨텍스트 생성 및 저장
            project_info = await self._analyze_project_structure(project_dir)
            project_context = ProjectContext(
                project_id=project_name,
                project_name=project_name,
                description=project_info.get('description', f'자동 인덱싱된 프로젝트: {project_name}'),
                tech_stack=project_info.get('tech_stack', []),
                file_patterns=project_info.get('file_patterns', [])
            )
            
            # 프로젝트 컨텍스트를 벡터 DB에 저장
            await self.vector_service.store_project_context(project_context)
            logger.info(f"프로젝트 컨텍스트 저장 완료: {project_name}")
            
            # 소스 파일 수집
            source_files = []
            for root, dirs, files in os.walk(project_dir):
                # 무시할 디렉토리 제외
                dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # 파일 필터링
                    if self._should_index_file(file_path):
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
                    
                    if self._should_index_file(file_path):
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
    
    def _should_index_file(self, file_path: Path) -> bool:
        """파일을 인덱싱해야 하는지 확인"""
        try:
            # 확장자 확인
            if file_path.suffix.lower() not in self.supported_extensions:
                return False
            
            # 파일명 패턴 확인
            file_name = file_path.name.lower()
            for pattern in self.ignore_file_patterns:
                if file_name.endswith(pattern.replace('*', '')):
                    return False
            
            # 숨김 파일 확인
            if file_name.startswith('.'):
                return False
            
            # 파일 크기 확인 (너무 큰 파일은 제외, 예: 10MB)
            try:
                if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
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
    
    async def get_status(self) -> Dict[str, Any]:
        """자동 인덱싱 서비스 상태 조회"""
        return {
            "is_running": self.is_running,
            "is_indexing": self.is_indexing,
            "indexed_files_count": len(self.indexed_files_cache),
            "scan_interval": self.scan_interval,
            "supported_extensions": list(self.supported_extensions),
            "host_projects_path": self.host_projects_path,
            "last_scan_time": max(
                (info.get('indexed_at', '') for info in self.indexed_files_cache.values()),
                default='없음'
            )
        }
    
    async def force_rescan(self, project_name: Optional[str] = None):
        """강제 재스캔 수행"""
        logger.info(f"강제 재스캔 시작 (프로젝트: {project_name or '전체'})")
        
        if project_name:
            # 특정 프로젝트만 재스캔
            project_path = Path(self.host_projects_path) / project_name
            if project_path.exists():
                await self._index_project_directory(project_path, project_name)
        else:
            # 전체 재스캔
            await self._perform_initial_scan()
        
        logger.info("강제 재스캔 완료") 

    async def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """프로젝트 구조 분석"""
        tech_stack = set()
        file_patterns = set()
        description = ""
        
        try:
            # README 파일에서 설명 추출
            for readme_name in ['README.md', 'README.txt', 'README.rst', 'README']:
                readme_path = project_path / readme_name
                if readme_path.exists():
                    try:
                        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # 첫 번째 단락을 설명으로 사용
                            lines = content.split('\n')
                            desc_lines = []
                            for line in lines[1:]:  # 제목 제외
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    desc_lines.append(line)
                                elif desc_lines:
                                    break
                            description = ' '.join(desc_lines)[:500]  # 500자 제한
                            break
                    except Exception as e:
                        logger.warning(f"README 파일 읽기 실패 {readme_path}: {e}")
            
            # 파일들을 스캔하여 기술 스택과 패턴 감지
            for root, dirs, files in os.walk(project_path):
                # 무시할 디렉토리 제외
                dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    extension = file_path.suffix.lower()
                    
                    # 파일 패턴 수집
                    if extension:
                        file_patterns.add(f"*{extension}")
                    
                    # 기술 스택 감지
                    if extension in ['.py']:
                        tech_stack.add('Python')
                    elif extension in ['.js', '.jsx']:
                        tech_stack.add('JavaScript')
                    elif extension in ['.ts', '.tsx']:
                        tech_stack.add('TypeScript')
                    elif extension in ['.java']:
                        tech_stack.add('Java')
                    elif extension in ['.cpp', '.c', '.h']:
                        tech_stack.add('C/C++')
                    elif extension in ['.cs']:
                        tech_stack.add('C#')
                    elif extension in ['.go']:
                        tech_stack.add('Go')
                    elif extension in ['.rs']:
                        tech_stack.add('Rust')
                    elif extension in ['.php']:
                        tech_stack.add('PHP')
                    elif extension in ['.rb']:
                        tech_stack.add('Ruby')
                    elif extension in ['.swift']:
                        tech_stack.add('Swift')
                    elif extension in ['.kt']:
                        tech_stack.add('Kotlin')
                    elif extension in ['.scala']:
                        tech_stack.add('Scala')
                    elif extension in ['.vue']:
                        tech_stack.add('Vue.js')
                    elif extension in ['.svelte']:
                        tech_stack.add('Svelte')
                    elif extension in ['.dart']:
                        tech_stack.add('Dart')
                    
                    # 특정 파일명으로 프레임워크/도구 감지
                    if file.lower() in ['package.json']:
                        tech_stack.add('Node.js')
                    elif file.lower() in ['requirements.txt', 'pyproject.toml', 'setup.py']:
                        tech_stack.add('Python')
                    elif file.lower() in ['composer.json']:
                        tech_stack.add('PHP')
                    elif file.lower() in ['cargo.toml']:
                        tech_stack.add('Rust')
                    elif file.lower() in ['pom.xml', 'build.gradle']:
                        tech_stack.add('Java')
                    elif file.lower() in ['dockerfile', 'docker-compose.yml', 'docker-compose.yaml']:
                        tech_stack.add('Docker')
                    elif file.lower() in ['next.config.js']:
                        tech_stack.add('Next.js')
                    elif file.lower() in ['nuxt.config.js']:
                        tech_stack.add('Nuxt.js')
                    elif file.lower() in ['angular.json']:
                        tech_stack.add('Angular')
            
        except Exception as e:
            logger.error(f"프로젝트 구조 분석 중 오류: {e}")
        
        return {
            'description': description,
            'tech_stack': list(tech_stack),
            'file_patterns': list(file_patterns)
        } 