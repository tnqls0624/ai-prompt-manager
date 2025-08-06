import os
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from models.prompt_models import ProjectContext, PromptHistory, PromptType
from services.vector_service import VectorService
from config import settings
import hashlib
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class FastIndexingService:
    """고성능 병렬 파일 인덱싱 서비스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        
        # 성능 설정 (대폭 증가)
        self.max_concurrent_files = 200  # 50 → 200으로 증가
        self.batch_size = 500  # 100 → 500으로 증가
        self.chunk_size = 2048  # 1024 → 2048로 증가
        self.chunk_overlap = 200  # 100 → 200으로 증가
        
        # 지원하는 파일 확장자
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.clj',
            '.md', '.txt', '.rst', '.asciidoc', '.org',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.sql', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
            '.dart', '.r', '.hs', '.elm', '.xml', '.dockerfile', '.env'
        }
        
        # 무시할 디렉토리 (확장)
        self.ignore_directories = {
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
        
        # 무시할 파일
        self.ignore_files = {
            '.gitignore', '.dockerignore', '.env', '.env.local', '.env.production',
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'bun.lockb',
            'poetry.lock', 'Pipfile.lock', 'pdm.lock', 'requirements.txt',
            'composer.lock', 'Gemfile.lock', 'Cargo.lock', 'go.sum', 'go.mod',
            'mix.lock', 'pubspec.lock', '.DS_Store', 'Thumbs.db'
        }
        
        # 성능 최적화를 위한 캐시
        self.file_cache = {}  # 파일 해시 캐시
        
        # Thread/Process Pool (워커 증가)
        self.thread_executor = ThreadPoolExecutor(max_workers=8)  # 4 → 8로 증가
        
    async def index_project_files_fast(self, project_path: str, project_id: str) -> Dict[str, Any]:
        """🚀 고속 병렬 프로젝트 파일 인덱싱"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 고속 인덱싱 시작: {project_path}")
            
            project_path = Path(project_path).resolve()
            
            if not project_path.exists():
                return {
                    "success": False,
                    "error": f"프로젝트 경로가 존재하지 않습니다: {project_path}"
                }
            
            # 1. 프로젝트 정보 수집 (병렬)
            project_info = await self._analyze_project_structure_fast(project_path)
            
            # 2. 프로젝트 컨텍스트 생성 및 저장
            project_context = ProjectContext(
                project_id=project_id,
                project_name=project_path.name,
                description=project_info.get('description', ''),
                tech_stack=project_info.get('tech_stack', []),
                file_patterns=project_info.get('file_patterns', [])
            )
            
            await self.vector_service.store_project_context(project_context)
            
            # 3. 파일 목록 수집 (Thread Pool 사용)
            logger.info("📂 파일 스캔 중...")
            loop = asyncio.get_event_loop()
            file_paths = await loop.run_in_executor(
                self.thread_executor, 
                self._scan_files_fast, 
                project_path
            )
            
            logger.info(f"📋 스캔 완료: {len(file_paths)}개 파일 발견")
            
            # 4. 병렬 파일 처리 (Semaphore로 동시성 제어)
            semaphore = asyncio.Semaphore(self.max_concurrent_files)
            indexed_files = []
            failed_files = []
            
            # 진행률 추적
            total_files = len(file_paths)
            processed_count = 0
            
            async def process_file_with_semaphore(file_path):
                nonlocal processed_count
                async with semaphore:
                    try:
                        await self._index_single_file_fast(file_path, project_id)
                        processed_count += 1
                        
                        # 진행률 로그 (10% 단위)
                        if processed_count % max(1, total_files // 10) == 0:
                            progress = (processed_count / total_files) * 100
                            logger.info(f"📊 진행률: {progress:.1f}% ({processed_count}/{total_files})")
                        
                        return str(file_path.relative_to(project_path))
                    except Exception as e:
                        logger.warning(f"파일 인덱싱 실패 {file_path}: {e}")
                        failed_files.append(str(file_path))
                        return None
            
            # 모든 파일을 병렬로 처리
            logger.info("⚡ 병렬 인덱싱 시작...")
            
            # 배치 단위로 처리하여 메모리 사용량 제어
            batch_size = 50  # 한 번에 처리할 파일 수
            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                logger.info(f"🔄 배치 {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size} 처리 중...")
                
                try:
                    batch_results = await asyncio.gather(
                        *[process_file_with_semaphore(fp) for fp in batch_files],
                        return_exceptions=True
                    )
                    
                    # 성공한 파일들만 수집
                    for result in batch_results:
                        if result is not None and not isinstance(result, Exception):
                            indexed_files.append(result)
                            
                except Exception as e:
                    logger.error(f"배치 처리 중 오류: {e}")
                    # 개별 파일로 재시도
                    for fp in batch_files:
                        try:
                            result = await process_file_with_semaphore(fp)
                            if result is not None:
                                indexed_files.append(result)
                        except Exception as file_error:
                            logger.warning(f"개별 파일 처리 실패 {fp}: {file_error}")
                            failed_files.append(str(fp))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                "success": True,
                "project_id": project_id,
                "project_name": project_context.project_name,
                "project_path": str(project_path),
                "total_files_found": len(file_paths),
                "indexed_files_count": len(indexed_files),
                "failed_files_count": len(failed_files),
                "processing_time_seconds": round(processing_time, 2),
                "files_per_second": round(len(indexed_files) / processing_time if processing_time > 0 else 0, 2),
                "tech_stack": project_context.tech_stack,
                "file_patterns": project_context.file_patterns,
                "indexed_files": indexed_files[:20],  # 처음 20개만 표시
                "failed_files": failed_files[:10] if failed_files else []
            }
            
            logger.info(f"✅ 고속 인덱싱 완료: {len(indexed_files)}개 파일 ({processing_time:.2f}초, {result['files_per_second']:.1f} files/sec)")
            return result
            
        except Exception as e:
            logger.error(f"고속 인덱싱 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }
    
    async def _analyze_project_structure_fast(self, project_path: Path) -> Dict[str, Any]:
        """🔍 병렬 프로젝트 구조 분석"""
        tech_stack = set()
        file_patterns = set()
        description = ""
        
        # README 파일들을 병렬로 처리
        readme_files = ['README.md', 'README.txt', 'README.rst', 'README']
        
        async def read_readme(filename):
            readme_path = project_path / filename
            if readme_path.exists():
                try:
                    async with aiofiles.open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                        lines = content.split('\n')
                        desc_lines = []
                        for line in lines[1:]:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                desc_lines.append(line)
                            elif desc_lines:
                                break
                        return ' '.join(desc_lines)[:500]
                except Exception as e:
                    logger.warning(f"README 파일 읽기 실패 {readme_path}: {e}")
            return None
        
        # README 파일들 병렬 읽기
        readme_results = await asyncio.gather(
            *[read_readme(filename) for filename in readme_files],
            return_exceptions=True
        )
        
        # 첫 번째 성공한 README 사용
        for result in readme_results:
            if result and not isinstance(result, Exception):
                description = result
                break
        
        # 설정 파일들 병렬 확인
        config_files = {
            'package.json': 'Node.js',
            'requirements.txt': 'Python',
            'Cargo.toml': 'Rust',
            'go.mod': 'Go',
            'pom.xml': 'Java/Maven',
            'build.gradle': 'Java/Gradle',
            'next.config.js': 'Next.js',
            'nuxt.config.js': 'Nuxt.js',
            'composer.json': 'PHP',
            'Gemfile': 'Ruby'
        }
        
        for config_file, tech in config_files.items():
            if (project_path / config_file).exists():
                tech_stack.add(tech)
        
        # 파일 확장자 분석 (샘플링으로 빠르게)
        sample_files = list(self._scan_files_fast(project_path))[:100]  # 처음 100개만 샘플링
        
        for file_path in sample_files:
            ext = file_path.suffix.lower()
            file_patterns.add(f"*{ext}")
            
            # 기술 스택 추정
            tech_mapping = {
                '.py': 'Python',
                '.js': 'JavaScript', '.jsx': 'JavaScript',
                '.ts': 'TypeScript', '.tsx': 'TypeScript',
                '.java': 'Java',
                '.cpp': 'C/C++', '.c': 'C/C++',
                '.cs': 'C#',
                '.go': 'Go',
                '.rs': 'Rust',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.swift': 'Swift',
                '.kt': 'Kotlin',
                '.vue': 'Vue.js',
                '.svelte': 'Svelte'
            }
            
            if ext in tech_mapping:
                tech_stack.add(tech_mapping[ext])
        
        return {
            'description': description,
            'tech_stack': sorted(list(tech_stack)),
            'file_patterns': sorted(list(file_patterns))
        }
    
    def _scan_files_fast(self, project_path: Path) -> List[Path]:
        """📂 고속 파일 스캔 (Thread Pool에서 실행)"""
        files = []
        
        for root, dirs, filenames in os.walk(project_path):
            # 무시할 디렉토리 제거
            dirs[:] = [d for d in dirs if d not in self.ignore_directories]
            
            for filename in filenames:
                if filename in self.ignore_files:
                    continue
                    
                file_path = Path(root) / filename
                
                # 파일 크기 체크
                try:
                    if file_path.stat().st_size > self.max_file_size:
                        continue
                except:
                    continue
                
                # 지원하는 확장자만 처리
                if file_path.suffix.lower() in self.supported_extensions:
                    files.append(file_path)
        
        return files
    
    async def _index_single_file_fast(self, file_path: Path, project_id: str):
        """⚡ 고속 단일 파일 인덱싱"""
        max_retries = 3
        retry_delay = 1  # 초
        
        for attempt in range(max_retries):
            try:
                # 파일 해시 계산 (캐싱용)
                file_stat = file_path.stat()
                file_key = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
                file_hash = hashlib.md5(file_key.encode()).hexdigest()
                
                # 캐시 확인 (이미 처리된 파일 스킵)
                if file_hash in self.file_cache:
                    return
                
                # 비동기 파일 읽기
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                
                # 빈 파일이나 너무 작은 파일 제외
                if len(content.strip()) < 20:
                    return
                
                # 더 큰 청크 크기로 처리 (API 호출 줄이기)
                if len(content) > self.chunk_size * 3:  # 6KB 이상이면 청킹
                    chunks = await self._chunk_file_content_fast(content, file_path)
                    
                    # 배치로 청크들 처리
                    for i in range(0, len(chunks), self.batch_size):
                        batch_chunks = chunks[i:i + self.batch_size]
                        
                        # 배치 처리 시 재시도 로직
                        for batch_attempt in range(max_retries):
                            try:
                                await asyncio.gather(*[
                                    self._store_file_chunk_fast(chunk, file_path, project_id, i + j)
                                    for j, chunk in enumerate(batch_chunks)
                                ])
                                break  # 성공하면 재시도 루프 종료
                            except Exception as batch_error:
                                if batch_attempt == max_retries - 1:
                                    raise batch_error
                                logger.warning(f"배치 저장 실패 (시도 {batch_attempt + 1}/{max_retries}): {batch_error}")
                                await asyncio.sleep(retry_delay * (batch_attempt + 1))
                else:
                    await self._store_file_chunk_fast(content, file_path, project_id, 0)
                
                # 캐시에 추가
                self.file_cache[file_hash] = file_path.name
                return  # 성공하면 재시도 루프 종료
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"고속 파일 인덱싱 최종 실패 {file_path}: {e}")
                    raise
                else:
                    logger.warning(f"고속 파일 인덱싱 실패 (시도 {attempt + 1}/{max_retries}) {file_path}: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # 지수 백오프
    
    async def _chunk_file_content_fast(self, content: str, file_path: Path) -> List[str]:
        """⚡ 고속 파일 청킹"""
        if self.vector_service.text_splitter:
            try:
                # LangChain 청킹 (더 큰 청크 크기)
                self.vector_service.text_splitter.chunk_size = self.chunk_size
                self.vector_service.text_splitter.chunk_overlap = self.chunk_overlap  # 더 큰 오버랩
                chunks = self.vector_service.text_splitter.split_text(content)
                return chunks
            except Exception as e:
                logger.warning(f"텍스트 청킹 실패 {file_path}: {e}")
        
        # 기본 청킹 (더 큰 청크)
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size > self.chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    async def _store_file_chunk_fast(self, content: str, file_path: Path, project_id: str, chunk_index: int):
        """⚡ 고속 파일 청크 저장"""
        # 고유 ID 생성
        file_hash = hashlib.md5(f"{file_path}_{chunk_index}".encode()).hexdigest()
        chunk_id = f"{project_id}_file_{file_hash}"
        
        # 메타데이터 구성
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "chunk_index": chunk_index,
            "file_type": self._detect_file_type(file_path),
            "is_file_content": True
        }
        
        # PromptHistory 객체 생성
        prompt_history = PromptHistory(
            id=chunk_id,
            project_id=project_id,
            content=content,
            prompt_type=PromptType.SYSTEM_PROMPT,
            metadata=metadata,
            created_at=datetime.now()
        )
        
        await self.vector_service.store_prompt_history(prompt_history)
    
    def _detect_file_type(self, file_path: Path) -> str:
        """📋 파일 타입 감지"""
        ext = file_path.suffix.lower()
        
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala'}
        doc_extensions = {'.md', '.txt', '.rst', '.asciidoc'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        
        if ext in code_extensions:
            return "code"
        elif ext in doc_extensions:
            return "documentation"
        elif ext in config_extensions:
            return "configuration"
        else:
            return "other"
    
    async def batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """🔥 배치 임베딩 생성 (고성능 최적화)"""
        try:
            if not texts:
                return []
                
            if not self.vector_service.embeddings:
                # 폴백: 개별 임베딩 생성 (병렬 처리)
                semaphore = asyncio.Semaphore(20)  # 동시성 제어
                
                async def embed_with_semaphore(text):
                    async with semaphore:
                        return await self.vector_service._generate_embedding(text)
                
                return await asyncio.gather(*[
                    embed_with_semaphore(text) for text in texts
                ])
            
            # 대용량 배치를 작은 청크로 분할하여 처리
            chunk_size = 100  # 한 번에 처리할 텍스트 수
            all_embeddings = []
            
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i + chunk_size]
                
                # DeepSeek 임베딩 배치 처리 (이미 내부적으로 최적화됨)
                chunk_embeddings = await self.vector_service.embeddings.aembed_documents(chunk_texts)
                all_embeddings.extend(chunk_embeddings)
                
                # 메모리 압박 방지를 위한 작은 대기
                if i % (chunk_size * 5) == 0 and i > 0:
                    await asyncio.sleep(0.01)  # 10ms 대기
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"배치 임베딩 실패: {e}")
            # 폴백: 더미 임베딩 반환
            return [[0.0] * 768] * len(texts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 성능 통계"""
        return {
            "max_concurrent_files": self.max_concurrent_files,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "cache_size": len(self.file_cache),
            "supported_extensions": len(self.supported_extensions)
        }
    
    def __del__(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'thread_executor'):
                self.thread_executor.shutdown(wait=False, cancel_futures=True)
                logger.debug("ThreadPoolExecutor 정리 완료")
        except Exception as e:
            logger.warning(f"리소스 정리 중 오류: {e}") 