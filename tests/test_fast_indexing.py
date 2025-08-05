import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from services.fast_indexing_service import FastIndexingService
from services.vector_service import VectorService
from services.file_indexing_service import FileIndexingService

@pytest.fixture
def vector_service():
    """Vector service fixture"""
    return VectorService()

@pytest.fixture
def fast_indexing_service(vector_service):
    """Fast indexing service fixture"""
    return FastIndexingService(vector_service)

@pytest.fixture
def regular_indexing_service(vector_service):
    """Regular indexing service fixture"""
    return FileIndexingService(vector_service)

@pytest.fixture
def sample_project():
    """샘플 프로젝트 생성"""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # 다양한 파일들 생성
        files_to_create = [
            ("main.py", """
import os
import sys
from typing import List, Dict, Any

def main():
    print("Hello, World!")
    data = {"key": "value"}
    return data

class ExampleClass:
    def __init__(self, name: str):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    main()
"""),
            ("utils.py", """
import json
import hashlib
from datetime import datetime

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def get_timestamp() -> str:
    return datetime.now().isoformat()

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
"""),
            ("config.json", """
{
    "app_name": "test_app",
    "version": "1.0.0",
    "debug": true,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "test_db"
    }
}
"""),
            ("README.md", """
# Test Project

This is a sample project for testing the fast indexing service.

## Features

- Fast file processing
- Parallel execution
- Smart caching

## Usage

```python
from fast_indexing import FastIndexer
indexer = FastIndexer()
result = await indexer.index_project("/path/to/project")
```
"""),
            ("package.json", """
{
    "name": "test-project",
    "version": "1.0.0",
    "description": "A test project",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "test": "jest"
    },
    "dependencies": {
        "express": "^4.18.0",
        "lodash": "^4.17.21"
    }
}
"""),
            ("server.js", """
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
    res.json({ message: 'Hello World!' });
});

app.get('/api/users', (req, res) => {
    const users = [
        { id: 1, name: 'John' },
        { id: 2, name: 'Jane' }
    ];
    res.json(users);
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
"""),
            ("styles.css", """
body {
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
}
"""),
            ("test_file.txt", """
This is a test file with some content.
It has multiple lines.
Each line contains different information.
This is used for testing file indexing performance.
The quick brown fox jumps over the lazy dog.
"""),
        ]
        
        # 파일들 생성
        for filename, content in files_to_create:
            file_path = project_path / filename
            file_path.write_text(content)
        
        # 하위 디렉토리도 생성
        src_dir = project_path / "src"
        src_dir.mkdir()
        
        (src_dir / "app.py").write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
""")
        
        yield project_path

class TestFastIndexing:
    """고속 인덱싱 성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_fast_indexing_performance(self, fast_indexing_service, sample_project):
        """고속 인덱싱 성능 테스트"""
        project_id = "test_fast_indexing"
        
        start_time = time.time()
        result = await fast_indexing_service.index_project_files_fast(
            str(sample_project), 
            project_id
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 결과 검증
        assert result["success"] == True
        assert result["indexed_files_count"] > 0
        assert result["processing_time_seconds"] > 0
        assert result["files_per_second"] > 0
        
        print(f"\n🚀 고속 인덱싱 결과:")
        print(f"   - 처리된 파일 수: {result['indexed_files_count']}")
        print(f"   - 처리 시간: {processing_time:.2f}초")
        print(f"   - 처리 속도: {result['files_per_second']:.1f} files/sec")
        print(f"   - 실패한 파일: {result.get('failed_files_count', 0)}")
    
    @pytest.mark.asyncio
    async def test_regular_vs_fast_comparison(self, regular_indexing_service, fast_indexing_service, sample_project):
        """일반 인덱싱 vs 고속 인덱싱 성능 비교"""
        project_id_regular = "test_regular"
        project_id_fast = "test_fast"
        
        # 일반 인덱싱
        print("\n📋 일반 인덱싱 테스트...")
        start_regular = time.time()
        regular_result = await regular_indexing_service.index_project_files(
            str(sample_project), 
            project_id_regular
        )
        end_regular = time.time()
        regular_time = end_regular - start_regular
        
        # 고속 인덱싱
        print("\n🚀 고속 인덱싱 테스트...")
        start_fast = time.time()
        fast_result = await fast_indexing_service.index_project_files_fast(
            str(sample_project), 
            project_id_fast
        )
        end_fast = time.time()
        fast_time = end_fast - start_fast
        
        # 성능 비교
        speedup = regular_time / fast_time if fast_time > 0 else 0
        
        print(f"\n📊 성능 비교 결과:")
        print(f"   일반 인덱싱: {regular_time:.2f}초 ({regular_result.get('indexed_files_count', 0)}개 파일)")
        print(f"   고속 인덱싱: {fast_time:.2f}초 ({fast_result.get('indexed_files_count', 0)}개 파일)")
        print(f"   성능 향상: {speedup:.1f}배 빠름")
        
        # 검증
        assert regular_result["success"] == True
        assert fast_result["success"] == True
        assert speedup >= 1.0  # 최소 1배 이상 빨라야 함
    
    @pytest.mark.asyncio
    async def test_file_caching(self, fast_indexing_service, sample_project):
        """파일 캐싱 기능 테스트"""
        project_id = "test_caching"
        
        # 첫 번째 인덱싱
        print("\n💾 첫 번째 인덱싱 (캐시 없음)...")
        start_first = time.time()
        result1 = await fast_indexing_service.index_project_files_fast(
            str(sample_project), 
            project_id
        )
        end_first = time.time()
        first_time = end_first - start_first
        
        # 두 번째 인덱싱 (캐시 사용)
        print("\n⚡ 두 번째 인덱싱 (캐시 사용)...")
        start_second = time.time()
        result2 = await fast_indexing_service.index_project_files_fast(
            str(sample_project), 
            project_id
        )
        end_second = time.time()
        second_time = end_second - start_second
        
        # 캐시 효과 확인
        cache_speedup = first_time / second_time if second_time > 0 else 0
        
        print(f"\n💾 캐싱 효과:")
        print(f"   첫 번째: {first_time:.2f}초")
        print(f"   두 번째: {second_time:.2f}초") 
        print(f"   캐시 효과: {cache_speedup:.1f}배 빠름")
        
        # 검증
        assert result1["success"] == True
        assert result2["success"] == True
        assert cache_speedup >= 1.0  # 캐시로 인한 성능 향상
    
    @pytest.mark.asyncio
    async def test_performance_stats(self, fast_indexing_service):
        """성능 통계 테스트"""
        stats = fast_indexing_service.get_performance_stats()
        
        print(f"\n📊 성능 설정 통계:")
        print(f"   - 최대 동시 파일 수: {stats['max_concurrent_files']}")
        print(f"   - 배치 크기: {stats['batch_size']}")
        print(f"   - 청크 크기: {stats['chunk_size']}")
        print(f"   - 캐시 크기: {stats['cache_size']}")
        print(f"   - 지원 확장자 수: {stats['supported_extensions']}")
        
        # 검증
        assert stats["max_concurrent_files"] > 0
        assert stats["batch_size"] > 0
        assert stats["chunk_size"] > 0
        assert stats["supported_extensions"] > 0
    
    @pytest.mark.asyncio
    async def test_project_structure_analysis(self, fast_indexing_service, sample_project):
        """프로젝트 구조 분석 테스트"""
        project_info = await fast_indexing_service._analyze_project_structure_fast(sample_project)
        
        print(f"\n🔍 프로젝트 구조 분석:")
        print(f"   - 기술 스택: {project_info['tech_stack']}")
        print(f"   - 파일 패턴: {project_info['file_patterns']}")
        print(f"   - 설명: {project_info['description'][:100]}...")
        
        # 검증
        assert len(project_info['tech_stack']) > 0
        assert len(project_info['file_patterns']) > 0
        assert 'Python' in project_info['tech_stack']
        assert 'JavaScript' in project_info['tech_stack']
    
    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, fast_indexing_service, sample_project):
        """동시 파일 처리 테스트"""
        project_id = "test_concurrent"
        
        # 동시성 설정 확인
        original_max_concurrent = fast_indexing_service.max_concurrent_files
        
        # 높은 동시성으로 테스트
        fast_indexing_service.max_concurrent_files = 50
        
        start_time = time.time()
        result = await fast_indexing_service.index_project_files_fast(
            str(sample_project), 
            project_id
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"\n⚡ 고동시성 처리 결과:")
        print(f"   - 동시 처리 수: {fast_indexing_service.max_concurrent_files}")
        print(f"   - 처리 시간: {processing_time:.2f}초")
        print(f"   - 처리 속도: {result.get('files_per_second', 0):.1f} files/sec")
        
        # 설정 복원
        fast_indexing_service.max_concurrent_files = original_max_concurrent
        
        # 검증
        assert result["success"] == True
        assert result["indexed_files_count"] > 0

if __name__ == "__main__":
    # 직접 실행 시 성능 테스트 데모
    async def demo():
        print("🚀 고속 인덱싱 성능 데모")
        print("=" * 50)
        
        vector_service = VectorService()
        fast_service = FastIndexingService(vector_service)
        
        # 현재 프로젝트 인덱싱
        current_project = Path.cwd()
        result = await fast_service.index_project_files_fast(str(current_project), "demo")
        
        print(f"\n✅ 데모 결과:")
        print(f"   프로젝트: {current_project.name}")
        print(f"   파일 수: {result.get('indexed_files_count', 0)}")
        print(f"   처리 시간: {result.get('processing_time_seconds', 0):.2f}초")
        print(f"   처리 속도: {result.get('files_per_second', 0):.1f} files/sec")
    
    asyncio.run(demo()) 