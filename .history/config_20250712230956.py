import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings:
    """애플리케이션 설정"""
    
    # 🔐 OpenAI 설정 (환경변수에서만 가져오기)
    openai_api_key: Optional[str] = None
    
    # 📊 벡터 DB 설정 (Docker 컨테이너 내부 경로)
    chroma_db_path: str = "/data"
    
    # 🌐 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000  # FastMCP 서버 포트
    
    # 📝 로깅 설정
    log_level: str = "INFO"
    log_dir: str = "/app/logs"  # Docker 컨테이너 내부 로그 경로
    
    # 🚀 MCP 설정
    mcp_server_name: str = "FastMCP Prompt Enhancement Server"
    mcp_version: str = "2.0.0"
    
    # 🔍 ChromaDB 설정
    chroma_host: str = "chromadb"  # Docker Compose 서비스명
    chroma_port: int = 8000       # ChromaDB 포트
    chroma_collection_name: str = "prompts"
    
    # 🧠 AI 설정
    max_context_length: int = 5
    similarity_threshold: float = 0.7
    
    # 📈 분석 설정
    enable_advanced_analytics: bool = True
    clustering_algorithm: str = "kmeans"
    max_clusters: int = 10
    
    # ⚡ 성능 최적화 설정
    # 캐싱 설정
    enable_search_cache: bool = True
    cache_ttl_seconds: int = 300  # 5분
    max_cache_size: int = 1000    # 최대 캐시 항목 수
    
    # 비동기 처리 설정
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    
    # 임베딩 배치 처리 설정
    embedding_batch_size: int = 50
    embedding_parallel_workers: int = 4
    
    # 파일 인덱싱 최적화
    max_file_size_mb: int = 10
    parallel_file_processing: bool = True
    chunk_processing_batch_size: int = 100
    
    # 데이터베이스 연결 풀링
    db_connection_pool_size: int = 10
    db_connection_timeout: int = 5
    
    # 메모리 관리
    enable_memory_optimization: bool = True
    garbage_collection_threshold: int = 1000  # 처리된 요청 수
    
    # 모니터링 설정
    enable_performance_monitoring: bool = True
    slow_query_threshold_seconds: float = 1.0
    
    # API 속도 제한
    rate_limit_per_minute: int = 100
    rate_limit_burst_size: int = 10
    
    # 텍스트 처리 최적화
    text_chunk_size: int = 1000
    text_chunk_overlap: int = 200
    max_text_length: int = 50000
    
    # 검색 최적화
    semantic_search_weight: float = 0.7
    keyword_search_weight: float = 0.3
    time_decay_factor: float = 0.1
    complexity_weight: float = 0.1
    
    # 리소스 제한
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: int = 80
    
    # 🔄 피드백 시스템 설정
    feedback_learning_rate: float = 0.1  # 학습률 (0.0 ~ 1.0)
    feedback_decay_rate: float = 0.95  # 시간에 따른 피드백 가중치 감소
    min_feedback_count: int = 3  # 최소 피드백 수
    feedback_confidence_threshold: float = 0.7  # 피드백 신뢰도 임계값
    enable_feedback_learning: bool = True  # 피드백 학습 활성화
    feedback_retention_days: int = 90  # 피드백 보존 기간
    
    # 프로젝트 선택 설정
    auto_indexing_enabled: bool = True
    auto_indexing_interval: int = 300  # 5분
    selective_indexing_enabled: bool = True
    project_whitelist_file: str = "./data/project_whitelist.json"
    host_projects_path: str = "/host_projects"
    max_file_size: int = 10485760  # 10MB
    max_files_per_project: int = 1000
    max_workers: int = 10
    batch_size: int = 50
    log_file: str = "./logs/mcp_server.log"
    file_watcher_enabled: bool = False
    file_watcher_debounce: float = 2.0
    vector_search_limit: int = 10
    backup_enabled: bool = True
    backup_interval: int = 86400  # 1일
    backup_retention_days: int = 30

    def __init__(self):
        # 기본 설정
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "3000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # OpenAI API 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # ChromaDB 설정
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma")
        self.chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "prompt_history")
        
        # 자동 인덱싱 설정
        self.auto_indexing_enabled = os.getenv("AUTO_INDEXING_ENABLED", "true").lower() == "true"
        self.auto_indexing_interval = int(os.getenv("AUTO_INDEXING_INTERVAL", "300"))  # 5분
        self.selective_indexing_enabled = os.getenv("SELECTIVE_INDEXING_ENABLED", "true").lower() == "true"
        self.project_whitelist_file = os.getenv("PROJECT_WHITELIST_FILE", "./data/project_whitelist.json")
        
        # 프로젝트 경로 설정
        self.host_projects_path = os.getenv("HOST_PROJECTS_PATH", "/host_projects")
        
        # 인덱싱 제한 설정
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.max_files_per_project = int(os.getenv("MAX_FILES_PER_PROJECT", "1000"))
        
        # 성능 설정
        self.max_workers = int(os.getenv("MAX_WORKERS", "10"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "50"))
        
        # 로깅 설정
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "./logs/mcp_server.log")
        
        # 파일 감시 설정
        self.file_watcher_enabled = os.getenv("FILE_WATCHER_ENABLED", "false").lower() == "true"
        self.file_watcher_debounce = float(os.getenv("FILE_WATCHER_DEBOUNCE", "2.0"))
        
        # 벡터 검색 설정
        self.vector_search_limit = int(os.getenv("VECTOR_SEARCH_LIMIT", "10"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # 백업 설정
        self.backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        self.backup_interval = int(os.getenv("BACKUP_INTERVAL", "86400"))  # 1일
        self.backup_retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

settings = Settings()

# 호환성을 위한 Config 클래스 생성
class Config:
    """정적 설정 클래스 (호환성 유지)"""
    
    @property
    def FEEDBACK_LEARNING_RATE(self) -> float:
        return settings.feedback_learning_rate
        
    @property
    def FEEDBACK_DECAY_RATE(self) -> float:
        return settings.feedback_decay_rate
        
    @property
    def MIN_FEEDBACK_COUNT(self) -> int:
        return settings.min_feedback_count
        
    @property
    def FEEDBACK_CONFIDENCE_THRESHOLD(self) -> float:
        return settings.feedback_confidence_threshold 