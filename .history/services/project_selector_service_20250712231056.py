"""
프로젝트 선택 및 화이트리스트 관리 서비스
사용자가 인덱싱할 프로젝트를 선택적으로 관리할 수 있는 기능 제공
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import aiofiles
from datetime import datetime

logger = logging.getLogger(__name__)

class ProjectSelectorService:
    """프로젝트 선택 및 화이트리스트 관리"""
    
    def __init__(self, whitelist_file: str = "./data/project_whitelist.json"):
        self.whitelist_file = whitelist_file
        self.selected_projects: Set[str] = set()
        self.project_configs: Dict[str, Dict[str, Any]] = {}
        
        # 데이터 디렉토리 생성
        os.makedirs(os.path.dirname(self.whitelist_file), exist_ok=True)
        
        logger.info(f"프로젝트 선택 서비스 초기화: {self.whitelist_file}")
    
    async def load_whitelist(self) -> bool:
        """화이트리스트 파일 로드"""
        try:
            if os.path.exists(self.whitelist_file):
                async with aiofiles.open(self.whitelist_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                    self.selected_projects = set(data.get('selected_projects', []))
                    self.project_configs = data.get('project_configs', {})
                    
                    logger.info(f"화이트리스트 로드 완료: {len(self.selected_projects)}개 프로젝트")
                    return True
            else:
                # 기본 화이트리스트 생성
                await self.save_whitelist()
                logger.info("기본 화이트리스트 파일 생성")
                return True
                
        except Exception as e:
            logger.error(f"화이트리스트 로드 실패: {e}")
            return False
    
    async def save_whitelist(self) -> bool:
        """화이트리스트 파일 저장"""
        try:
            data = {
                'selected_projects': list(self.selected_projects),
                'project_configs': self.project_configs,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            async with aiofiles.open(self.whitelist_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            
            logger.info(f"화이트리스트 저장 완료: {len(self.selected_projects)}개 프로젝트")
            return True
            
        except Exception as e:
            logger.error(f"화이트리스트 저장 실패: {e}")
            return False
    
    async def add_project(self, project_name: str, config: Dict[str, Any] = None) -> bool:
        """프로젝트를 화이트리스트에 추가"""
        try:
            self.selected_projects.add(project_name)
            
            if config is None:
                config = {}
            
            # 기본 설정 추가
            default_config = {
                'enabled': True,
                'auto_indexing': True,
                'file_watcher': False,
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'max_files': 1000,
                'include_extensions': ['.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.md'],
                'exclude_directories': ['node_modules', '.git', '__pycache__', 'dist', 'build'],
                'added_at': datetime.now().isoformat()
            }
            
            # 사용자 설정과 기본 설정 병합
            merged_config = {**default_config, **config}
            self.project_configs[project_name] = merged_config
            
            await self.save_whitelist()
            logger.info(f"프로젝트 추가: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 추가 실패 {project_name}: {e}")
            return False
    
    async def remove_project(self, project_name: str) -> bool:
        """프로젝트를 화이트리스트에서 제거"""
        try:
            if project_name in self.selected_projects:
                self.selected_projects.remove(project_name)
            
            if project_name in self.project_configs:
                del self.project_configs[project_name]
            
            await self.save_whitelist()
            logger.info(f"프로젝트 제거: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 제거 실패 {project_name}: {e}")
            return False
    
    async def update_project_config(self, project_name: str, config: Dict[str, Any]) -> bool:
        """프로젝트 설정 업데이트"""
        try:
            if project_name not in self.selected_projects:
                return False
            
            if project_name in self.project_configs:
                self.project_configs[project_name].update(config)
                self.project_configs[project_name]['updated_at'] = datetime.now().isoformat()
            else:
                self.project_configs[project_name] = config
            
            await self.save_whitelist()
            logger.info(f"프로젝트 설정 업데이트: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 설정 업데이트 실패 {project_name}: {e}")
            return False
    
    def is_project_selected(self, project_name: str) -> bool:
        """프로젝트가 선택되었는지 확인"""
        return project_name in self.selected_projects
    
    def get_project_config(self, project_name: str) -> Optional[Dict[str, Any]]:
        """프로젝트 설정 조회"""
        return self.project_configs.get(project_name)
    
    def get_selected_projects(self) -> List[str]:
        """선택된 프로젝트 목록 조회"""
        return list(self.selected_projects)
    
    def get_all_project_configs(self) -> Dict[str, Dict[str, Any]]:
        """모든 프로젝트 설정 조회"""
        return self.project_configs.copy()
    
    async def discover_available_projects(self, base_path: str = "/host_projects") -> List[Dict[str, Any]]:
        """사용 가능한 프로젝트 목록 탐지"""
        projects = []
        
        try:
            if not os.path.exists(base_path):
                return projects
            
            base_path = Path(base_path)
            
            for item in base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # 프로젝트 정보 수집
                    project_info = {
                        'name': item.name,
                        'path': str(item),
                        'selected': self.is_project_selected(item.name),
                        'size': await self._get_directory_size(item),
                        'file_count': await self._count_source_files(item),
                        'last_modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                        'tech_stack': await self._detect_tech_stack(item)
                    }
                    
                    projects.append(project_info)
            
            # 이름순으로 정렬
            projects.sort(key=lambda x: x['name'])
            
        except Exception as e:
            logger.error(f"프로젝트 탐지 중 오류: {e}")
        
        return projects
    
    async def _get_directory_size(self, path: Path) -> int:
        """디렉토리 크기 계산 (MB)"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(path):
                # 무시할 디렉토리 제외
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'dist', 'build']]
                
                for file in files:
                    try:
                        file_path = Path(root) / file
                        total_size += file_path.stat().st_size
                    except:
                        continue
            
            return total_size // (1024 * 1024)  # MB 단위
            
        except Exception as e:
            logger.warning(f"디렉토리 크기 계산 실패 {path}: {e}")
            return 0
    
    async def _count_source_files(self, path: Path) -> int:
        """소스 파일 개수 계산"""
        try:
            count = 0
            source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.md', '.json', '.yaml', '.yml'}
            
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'dist', 'build']]
                
                for file in files:
                    if Path(file).suffix.lower() in source_extensions:
                        count += 1
            
            return count
            
        except Exception as e:
            logger.warning(f"소스 파일 개수 계산 실패 {path}: {e}")
            return 0
    
    async def _detect_tech_stack(self, path: Path) -> List[str]:
        """기술 스택 탐지"""
        tech_stack = []
        
        try:
            # 파일 기반 탐지
            if (path / 'package.json').exists():
                tech_stack.append('Node.js')
            if (path / 'requirements.txt').exists():
                tech_stack.append('Python')
            if (path / 'Cargo.toml').exists():
                tech_stack.append('Rust')
            if (path / 'go.mod').exists():
                tech_stack.append('Go')
            if (path / 'pom.xml').exists():
                tech_stack.append('Java')
            if (path / 'Dockerfile').exists():
                tech_stack.append('Docker')
            
            # 확장자 기반 탐지
            extensions = set()
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
                
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext:
                        extensions.add(ext)
                
                # 너무 깊이 탐색하지 않도록 제한
                if len(Path(root).relative_to(path).parts) > 2:
                    dirs.clear()
            
            # 확장자로 기술 스택 추가
            if '.py' in extensions:
                tech_stack.append('Python')
            if '.js' in extensions or '.jsx' in extensions:
                tech_stack.append('JavaScript')
            if '.ts' in extensions or '.tsx' in extensions:
                tech_stack.append('TypeScript')
            if '.vue' in extensions:
                tech_stack.append('Vue.js')
            if '.svelte' in extensions:
                tech_stack.append('Svelte')
            
        except Exception as e:
            logger.warning(f"기술 스택 탐지 실패 {path}: {e}")
        
        return list(set(tech_stack))  # 중복 제거
    
    async def get_status(self) -> Dict[str, Any]:
        """프로젝트 선택 서비스 상태 조회"""
        return {
            'selected_projects_count': len(self.selected_projects),
            'selected_projects': list(self.selected_projects),
            'whitelist_file': self.whitelist_file,
            'last_updated': datetime.now().isoformat()
        } 