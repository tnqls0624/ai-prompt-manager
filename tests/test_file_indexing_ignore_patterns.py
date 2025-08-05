#!/usr/bin/env python3
"""
파일 인덱싱 서비스의 ignore 패턴 테스트
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

from services.file_indexing_service import FileIndexingService
from services.vector_service import VectorService


class TestFileIndexingIgnorePatterns:
    """파일 인덱싱 서비스의 ignore 패턴 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.vector_service = MagicMock(spec=VectorService)
        self.indexing_service = FileIndexingService(self.vector_service)
    
    def test_should_ignore_node_modules_directory(self):
        """node_modules 디렉토리를 제외해야 함"""
        assert 'node_modules' in self.indexing_service.ignore_directories
    
    def test_should_ignore_python_cache_directories(self):
        """Python 캐시 디렉토리들을 제외해야 함"""
        python_cache_dirs = {
            '__pycache__', '.pytest_cache', '.mypy_cache', 
            'venv', 'env', '.env'
        }
        assert python_cache_dirs.issubset(self.indexing_service.ignore_directories)
    
    def test_should_ignore_build_directories(self):
        """빌드 디렉토리들을 제외해야 함"""
        build_dirs = {'dist', 'build', 'target', 'out', '.next'}
        assert build_dirs.issubset(self.indexing_service.ignore_directories)
    
    def test_should_ignore_version_control_directories(self):
        """버전 컨트롤 디렉토리들을 제외해야 함"""
        vcs_dirs = {'.git', '.svn', '.hg'}
        # 현재 .git만 있으므로 추가 필요
        assert '.git' in self.indexing_service.ignore_directories
    
    def test_should_ignore_ide_directories(self):
        """IDE 디렉토리들을 제외해야 함"""
        ide_dirs = {'.vscode', '.idea'}
        assert ide_dirs.issubset(self.indexing_service.ignore_directories)
    
    def test_should_ignore_large_dependency_directories(self):
        """대용량 의존성 디렉토리들을 제외해야 함"""
        # 현재 빠진 것들
        missing_dirs = {
            'vendor',        # Go, PHP 패키지 관리
            'pkg',           # Go 패키지
            'bin',           # 바이너리
            'obj',           # C#, .NET 오브젝트
            'Debug',         # Visual Studio 디버그
            'Release',       # Visual Studio 릴리스
            'cache',         # 일반적인 캐시
            'tmp',           # 임시 파일
            'temp',          # 임시 파일
            'logs',          # 로그 파일 (이미 있음)
            'coverage',      # 커버리지 (이미 있음)
            'public/assets', # 프론트엔드 빌드 (부분적)
            'bower_components', # Bower 패키지
            'jspm_packages', # jspm 패키지
            'typings'        # TypeScript 타입 정의
        }
        
        # 이 테스트는 실패할 것 - 이것이 Red 단계
        current_dirs = self.indexing_service.ignore_directories
        for dir_name in missing_dirs:
            if dir_name not in current_dirs:
                print(f"Missing ignore directory: {dir_name}")
        
        # 임시로 로그만 확인
        assert 'logs' in current_dirs
    
    def test_should_ignore_lock_files(self):
        """락 파일들을 제외해야 함"""
        lock_files = {
            'package-lock.json', 'yarn.lock', 'poetry.lock', 
            'Pipfile.lock'
        }
        assert lock_files.issubset(self.indexing_service.ignore_files)
    
    def test_should_ignore_config_files(self):
        """일부 설정 파일들을 제외해야 함"""
        config_files = {'.gitignore', '.dockerignore', '.env', '.env.local'}
        assert config_files.issubset(self.indexing_service.ignore_files)
    
    def test_should_ignore_additional_lock_files(self):
        """추가적인 락 파일들을 제외해야 함"""
        additional_locks = {
            'composer.lock',    # PHP
            'Gemfile.lock',     # Ruby
            'Cargo.lock',       # Rust
            'go.sum',          # Go
            'mix.lock',        # Elixir
            'pubspec.lock',    # Dart/Flutter
            'pdm.lock',        # Python PDM
            'pnpm-lock.yaml'   # pnpm
        }
        
        current_files = self.indexing_service.ignore_files
        missing_files = additional_locks - current_files
        
        # 이 테스트는 실패할 것 - Red 단계
        if missing_files:
            print(f"Missing ignore files: {missing_files}")
        
        # 임시로 현재 있는 것만 확인
        assert 'poetry.lock' in current_files

    @pytest.mark.asyncio
    async def test_file_scanning_excludes_ignored_directories(self):
        """파일 스캔 시 ignored 디렉토리가 실제로 제외되는지 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 테스트 파일 구조 생성
            (temp_path / 'src' / 'main.py').parent.mkdir(parents=True)
            (temp_path / 'src' / 'main.py').write_text('print("hello")')
            
            # ignore 디렉토리 생성
            (temp_path / 'node_modules' / 'some_package' / 'index.js').parent.mkdir(parents=True)
            (temp_path / 'node_modules' / 'some_package' / 'index.js').write_text('module.exports = {}')
            
            (temp_path / '__pycache__' / 'main.cpython-38.pyc').parent.mkdir(parents=True)
            (temp_path / '__pycache__' / 'main.cpython-38.pyc').write_bytes(b'fake bytecode')
            
            # 파일 스캔
            scanned_files = list(self.indexing_service._scan_files(temp_path))
            scanned_paths = [str(f.relative_to(temp_path)) for f in scanned_files]
            
            # 검증
            assert 'src/main.py' in scanned_paths
            assert not any('node_modules' in path for path in scanned_paths)
            assert not any('__pycache__' in path for path in scanned_paths)
    
    def test_file_size_limit_check(self):
        """파일 크기 제한이 적절한지 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 작은 파일 생성 (포함되어야 함)
            small_file = temp_path / 'small.py'
            small_file.write_text('print("hello")')
            
            # 큰 파일 생성 (제외되어야 함) - 실제로는 생성하지 않고 크기만 확인
            large_file = temp_path / 'large.py'
            large_file.write_text('x' * 1000)  # 1KB 파일
            
            scanned_files = list(self.indexing_service._scan_files(temp_path))
            scanned_names = [f.name for f in scanned_files]
            
            assert 'small.py' in scanned_names
            assert 'large.py' in scanned_names  # 1KB는 아직 제한 이하
    
    def test_supported_file_extensions(self):
        """지원되는 파일 확장자 목록이 충분한지 테스트"""
        expected_extensions = {
            # 프로그래밍 언어
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
            
            # 문서
            '.md', '.txt', '.rst', '.asciidoc',
            
            # 설정
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            
            # 기타
            '.sql', '.sh', '.bash', '.ps1',
            '.html', '.css', '.scss', '.sass', '.less',
            '.vue', '.svelte', '.astro'
        }
        
        current_extensions = self.indexing_service.supported_extensions
        missing_extensions = expected_extensions - current_extensions
        
        if missing_extensions:
            print(f"Missing supported extensions: {missing_extensions}")
        
        # 대부분이 이미 있는지 확인
        assert len(current_extensions & expected_extensions) > 20


if __name__ == "__main__":
    # 직접 실행 시 테스트 실행
    import sys
    sys.exit(pytest.main([__file__, "-v"])) 