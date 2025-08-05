import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from models.prompt_models import ProjectContext, PromptHistory, PromptType
from services.vector_service import VectorService
from config import settings
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class FileIndexingService:
    """파일 인덱싱 서비스 - 프로젝트 파일들을 스캔하여 벡터 DB에 저장"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        
        # 지원하는 파일 확장자들
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
            '.md', '.txt', '.rst', '.asciidoc',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sql', '.sh', '.bash', '.ps1',
            '.html', '.css', '.scss', '.sass', '.less',
            '.vue', '.svelte', '.astro'
        }
        
        # 무시할 디렉토리들
        self.ignore_directories = {
            # JavaScript/Node.js 관련
            'node_modules', 'bower_components', 'jspm_packages', 'typings',
            
            # Python 관련
            '__pycache__', '.pytest_cache', '.mypy_cache', 'venv', 'env', '.env',
            
            # 버전 관리 시스템
            '.git', '.svn', '.hg',
            
            # IDE 관련 (중요: .history 포함)
            '.vscode', '.idea', '.history', '.vs',
            
            # 빌드 관련
            'dist', 'build', 'target', 'out', '.next', 'bin', 'obj',
            
            # 컴파일러별 빌드 디렉토리
            'Debug', 'Release',  # Visual Studio
            
            # 언어별 패키지 관리
            'vendor',  # Go, PHP, Ruby 등
            'pkg',     # Go 패키지
            '.gradle', '.maven',  # Java 빌드 도구
            
            # 캐시 및 임시 파일
            'cache', 'tmp', 'temp', 'coverage', 'logs', 'nyc_output', '.sass-cache',
            
            # 정적 파일 및 에셋 (일반적으로 불필요)
            'assets', 'public', 'static',
            
            # 데이터베이스 관련
            'chroma_db',
            
            # 기타 숨김 디렉토리
            '.DS_Store'
        }
        
        # 무시할 파일들
        self.ignore_files = {
            # 환경 및 설정 파일
            '.gitignore', '.dockerignore', '.env', '.env.local',
            
            # 패키지 매니저 락 파일들
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',  # JavaScript/Node.js
            'poetry.lock', 'Pipfile.lock', 'pdm.lock',           # Python
            'composer.lock',                                      # PHP
            'Gemfile.lock',                                       # Ruby
            'Cargo.lock',                                         # Rust
            'go.sum',                                             # Go
            'mix.lock',                                           # Elixir
            'pubspec.lock',                                       # Dart/Flutter
            
            # 기타 제외할 파일들
            # 'requirements.txt'  # Python 의존성 파일은 인덱싱 대상 (설정 파일)
        }
    
    async def index_project_files(self, project_path: str, project_id: str) -> Dict[str, Any]:
        """프로젝트 파일들을 인덱싱"""
        try:
            logger.info(f"프로젝트 파일 인덱싱 시작: {project_path}")
            
            project_path = Path(project_path).resolve()
            
            if not project_path.exists():
                return {
                    "success": False,
                    "error": f"프로젝트 경로가 존재하지 않습니다: {project_path}"
                }
            
            # 프로젝트 정보 수집
            project_info = await self._analyze_project_structure(project_path)
            
            # 프로젝트 컨텍스트 생성 및 저장
            project_context = ProjectContext(
                project_id=project_id,
                project_name=project_path.name,
                description=project_info.get('description', ''),
                tech_stack=project_info.get('tech_stack', []),
                file_patterns=project_info.get('file_patterns', [])
            )
            
            await self.vector_service.store_project_context(project_context)
            
            # 파일들 스캔 및 저장
            indexed_files = []
            file_count = 0
            
            for file_path in self._scan_files(project_path):
                try:
                    await self._index_single_file(file_path, project_id)
                    indexed_files.append(str(file_path.relative_to(project_path)))
                    file_count += 1
                    
                    # 50개마다 로그
                    if file_count % 50 == 0:
                        logger.info(f"인덱싱 진행 중: {file_count}개 파일 완료")
                        
                except Exception as e:
                    logger.warning(f"파일 인덱싱 실패 {file_path}: {e}")
                    continue
            
            result = {
                "success": True,
                "project_id": project_id,
                "project_name": project_context.project_name,
                "project_path": str(project_path),
                "indexed_files_count": len(indexed_files),
                "tech_stack": project_context.tech_stack,
                "file_patterns": project_context.file_patterns,
                "indexed_files": indexed_files[:20]  # 처음 20개만 반환
            }
            
            logger.info(f"프로젝트 인덱싱 완료: {len(indexed_files)}개 파일")
            return result
            
        except Exception as e:
            logger.error(f"프로젝트 인덱싱 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }
    
    async def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """프로젝트 구조 분석"""
        tech_stack = set()
        file_patterns = set()
        description = ""
        
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
        
        # 파일 확장자 및 기술 스택 분석
        for file_path in self._scan_files(project_path):
            ext = file_path.suffix.lower()
            file_patterns.add(f"*{ext}")
            
            # 기술 스택 추정
            if ext in ['.py']:
                tech_stack.add('Python')
            elif ext in ['.js', '.jsx']:
                tech_stack.add('JavaScript')
            elif ext in ['.ts', '.tsx']:
                tech_stack.add('TypeScript')
            elif ext in ['.java']:
                tech_stack.add('Java')
            elif ext in ['.cpp', '.c']:
                tech_stack.add('C/C++')
            elif ext in ['.cs']:
                tech_stack.add('C#')
            elif ext in ['.go']:
                tech_stack.add('Go')
            elif ext in ['.rs']:
                tech_stack.add('Rust')
            elif ext in ['.php']:
                tech_stack.add('PHP')
            elif ext in ['.rb']:
                tech_stack.add('Ruby')
            elif ext in ['.swift']:
                tech_stack.add('Swift')
            elif ext in ['.kt']:
                tech_stack.add('Kotlin')
            elif ext in ['.vue']:
                tech_stack.add('Vue.js')
            elif ext in ['.svelte']:
                tech_stack.add('Svelte')
        
        # 특정 파일들로 추가 기술 스택 감지
        if (project_path / 'package.json').exists():
            tech_stack.add('Node.js')
        if (project_path / 'requirements.txt').exists():
            tech_stack.add('Python')
        if (project_path / 'Cargo.toml').exists():
            tech_stack.add('Rust')
        if (project_path / 'go.mod').exists():
            tech_stack.add('Go')
        if (project_path / 'pom.xml').exists():
            tech_stack.add('Java/Maven')
        if (project_path / 'build.gradle').exists():
            tech_stack.add('Java/Gradle')
        if (project_path / 'next.config.js').exists():
            tech_stack.add('Next.js')
        if (project_path / 'nuxt.config.js').exists():
            tech_stack.add('Nuxt.js')
        
        return {
            'description': description,
            'tech_stack': sorted(list(tech_stack)),
            'file_patterns': sorted(list(file_patterns))
        }
    
    def _scan_files(self, project_path: Path):
        """프로젝트 파일들을 스캔 (주요 소스코드만)"""
        for root, dirs, files in os.walk(project_path):
            # 무시할 디렉토리 제거 + .으로 시작하는 모든 디렉토리 제외
            dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]
            
            # 현재 경로에 .으로 시작하는 디렉토리가 포함되어 있으면 건너뛰기 (.history/ 등)
            try:
                relative_root = Path(root).relative_to(project_path)
                if any(part.startswith('.') for part in relative_root.parts):
                    continue
            except ValueError:
                # relative_to 실패시 건너뛰기
                continue
            
            for file in files:
                if file in self.ignore_files:
                    continue
                
                # .으로 시작하는 숨김 파일 제외
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # 파일 크기 체크 (10MB 이상은 제외)
                try:
                    if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                        continue
                except:
                    continue
                
                # 지원하는 확장자만 처리 (주요 소스코드 파일)
                if file_path.suffix.lower() in self.supported_extensions:
                    yield file_path
    
    async def _index_single_file(self, file_path: Path, project_id: str):
        """단일 파일 인덱싱"""
        try:
            # 파일 내용 읽기
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 빈 파일이나 너무 작은 파일 제외
            if len(content.strip()) < 10:
                return
            
            # 파일이 너무 크면 청킹
            if len(content) > 8000:  # 8KB 이상이면 청킹
                chunks = await self._chunk_file_content(content, file_path)
                for i, chunk in enumerate(chunks):
                    await self._store_file_chunk(chunk, file_path, project_id, i)
            else:
                await self._store_file_chunk(content, file_path, project_id, 0)
                
        except Exception as e:
            logger.warning(f"파일 인덱싱 실패 {file_path}: {e}")
            raise
    
    async def _chunk_file_content(self, content: str, file_path: Path) -> List[str]:
        """파일 내용을 청킹"""
        if self.vector_service.text_splitter:
            try:
                chunks = self.vector_service.text_splitter.split_text(content)
                return chunks
            except Exception as e:
                logger.warning(f"텍스트 청킹 실패 {file_path}: {e}")
        
        # 기본 청킹 (줄 단위)
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size > 4000:  # 4KB씩 청킹
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    async def _store_file_chunk(self, content: str, file_path: Path, project_id: str, chunk_index: int):
        """파일 청크를 벡터 DB에 저장"""
        # 고유 ID 생성
        file_hash = hashlib.md5(f"{file_path}_{chunk_index}".encode()).hexdigest()
        chunk_id = f"{project_id}_file_{file_hash}"
        
        # 메타데이터 구성
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "chunk_index": chunk_index,
            "file_type": "code" if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.jsx', '.tsx'] else "documentation",
            "is_file_content": True  # 파일 내용임을 표시
        }
        
        # PromptHistory 객체 생성 (파일 내용을 프롬프트 히스토리로 저장)
        prompt_history = PromptHistory(
            id=chunk_id,
            project_id=project_id,
            content=content,
            prompt_type=PromptType.SYSTEM_PROMPT,  # 파일 내용은 시스템 프롬프트로 분류
            metadata=metadata,
            created_at=datetime.now()
        )
        
        await self.vector_service.store_prompt_history(prompt_history)

    async def store_file_content(self, filename: str, content: str, project_id: str, metadata: Dict[str, Any] = None) -> bool:
        """파일 내용을 직접 벡터 DB에 저장 (AutoIndexingService에서 사용)"""
        try:
            # 빈 파일이나 너무 작은 파일 제외
            if len(content.strip()) < 10:
                return False
            
            # 메타데이터 기본값 설정
            if metadata is None:
                metadata = {}
            
            # Path 객체 생성 (메타데이터에서 file_path 사용 또는 filename 사용)
            file_path_str = metadata.get('file_path', filename)
            file_path = Path(file_path_str) if file_path_str else Path(filename)
            
            # 파일이 너무 크면 청킹
            if len(content) > 8000:  # 8KB 이상이면 청킹
                chunks = await self._chunk_file_content(content, file_path)
                for i, chunk in enumerate(chunks):
                    await self._store_file_chunk_with_metadata(chunk, file_path, project_id, i, metadata)
            else:
                await self._store_file_chunk_with_metadata(content, file_path, project_id, 0, metadata)
            
            return True
                
        except Exception as e:
            logger.error(f"파일 내용 저장 실패 {filename}: {e}")
            return False

    async def _store_file_chunk_with_metadata(self, content: str, file_path: Path, project_id: str, chunk_index: int, additional_metadata: Dict[str, Any] = None):
        """파일 청크를 벡터 DB에 저장 (추가 메타데이터 포함)"""
        # 고유 ID 생성
        file_hash = hashlib.md5(f"{file_path}_{chunk_index}".encode()).hexdigest()
        chunk_id = f"{project_id}_file_{file_hash}"
        
        # 기본 메타데이터 구성
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "chunk_index": chunk_index,
            "file_type": "code" if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.jsx', '.tsx'] else "documentation",
            "is_file_content": True  # 파일 내용임을 표시
        }
        
        # 추가 메타데이터 병합
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # PromptHistory 객체 생성 (파일 내용을 프롬프트 히스토리로 저장)
        prompt_history = PromptHistory(
            id=chunk_id,
            project_id=project_id,
            content=content,
            prompt_type=PromptType.SYSTEM_PROMPT,  # 파일 내용은 시스템 프롬프트로 분류
            metadata=metadata,
            created_at=datetime.now()
        )
        
        await self.vector_service.store_prompt_history(prompt_history)

    async def update_project_index(self, project_path: str, project_id: str) -> Dict[str, Any]:
        """프로젝트 인덱스 업데이트 (기존 데이터 삭제 후 재인덱싱)"""
        try:
            # 기존 프로젝트 데이터 삭제
            await self.vector_service.delete_project_data(project_id)
            
            # 재인덱싱
            return await self.index_project_files(project_path, project_id)
        except Exception as e:
            logger.error(f"프로젝트 인덱스 업데이트 실패: {e}")
            return {
                "success": False,
                "error": f"프로젝트 인덱스 업데이트 실패: {str(e)}"
            } 