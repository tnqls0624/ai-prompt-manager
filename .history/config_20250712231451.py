import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings:
    """애플리케이션 설정"""
    
    def __init__(self):
        # 🔐 OpenAI 설정 (환경변수에서만 가져오기)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        
        # 📊 벡터 DB 설정 (Docker 컨테이너 내부 경로)
        self.chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "/data")
        
        # 🌐 서버 설정
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # 📝 로깅 설정
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_dir: str = os.getenv("LOG_DIR", "/data/logs")
        
        # 🚀 MCP 설정
        self.mcp_server_name: str = "FastMCP Prompt Enhancement Server"
        self.mcp_version: str = "2.0.0"
        
        # 🔍 ChromaDB 설정
        self.chroma_host: str = os.getenv("CHROMA_HOST", "chromadb")
        self.chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "prompts")
        
        # 🧠 AI 설정
        self.max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "5"))
        self.similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # 📈 분석 설정
        self.enable_advanced_analytics: bool = os.getenv("ENABLE_ADVANCED_ANALYTICS", "true").lower() == "true"
        self.clustering_algorithm: str = os.getenv("CLUSTERING_ALGORITHM", "kmeans")
        self.max_clusters: int = int(os.getenv("MAX_CLUSTERS", "10"))
        
        # ⚡ 성능 최적화 설정
        self.enable_search_cache: bool = os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true"
        self.cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
        self.max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
        
        # 비동기 처리 설정
        self.max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
        
        # 임베딩 배치 처리 설정
        self.embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
        self.embedding_parallel_workers: int = int(os.getenv("EMBEDDING_PARALLEL_WORKERS", "4"))
        
        # 파일 인덱싱 최적화
        self.max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
        self.parallel_file_processing: bool = os.getenv("PARALLEL_FILE_PROCESSING", "true").lower() == "true"
        self.chunk_processing_batch_size: int = int(os.getenv("CHUNK_PROCESSING_BATCH_SIZE", "100"))
        
        # 데이터베이스 연결 풀링
        self.db_connection_pool_size: int = int(os.getenv("DB_CONNECTION_POOL_SIZE", "10"))
        self.db_connection_timeout: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "5"))
        
        # 메모리 관리
        self.enable_memory_optimization: bool = os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true"
        self.garbage_collection_threshold: int = int(os.getenv("GARBAGE_COLLECTION_THRESHOLD", "1000"))
        
        # 모니터링 설정
        self.enable_performance_monitoring: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        self.slow_query_threshold_seconds: float = float(os.getenv("SLOW_QUERY_THRESHOLD_SECONDS", "1.0"))
        
        # API 속도 제한
        self.rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
        self.rate_limit_burst_size: int = int(os.getenv("RATE_LIMIT_BURST_SIZE", "10"))
        
        # 텍스트 처리 최적화
        self.text_chunk_size: int = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
        self.text_chunk_overlap: int = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))
        self.max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
        
        # 검색 최적화
        self.semantic_search_weight: float = float(os.getenv("SEMANTIC_SEARCH_WEIGHT", "0.7"))
        self.keyword_search_weight: float = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
        self.time_decay_factor: float = float(os.getenv("TIME_DECAY_FACTOR", "0.1"))
        self.complexity_weight: float = float(os.getenv("COMPLEXITY_WEIGHT", "0.1"))
        
        # 리소스 제한
        self.max_memory_usage_mb: int = int(os.getenv("MAX_MEMORY_USAGE_MB", "2048"))
        self.max_cpu_usage_percent: int = int(os.getenv("MAX_CPU_USAGE_PERCENT", "80"))
        
        # 🔄 피드백 시스템 설정
        self.feedback_learning_rate: float = float(os.getenv("FEEDBACK_LEARNING_RATE", "0.1"))
        self.feedback_decay_rate: float = float(os.getenv("FEEDBACK_DECAY_RATE", "0.95"))
        self.min_feedback_count: int = int(os.getenv("MIN_FEEDBACK_COUNT", "3"))
        self.feedback_confidence_threshold: float = float(os.getenv("FEEDBACK_CONFIDENCE_THRESHOLD", "0.7"))
        self.enable_feedback_learning: bool = os.getenv("ENABLE_FEEDBACK_LEARNING", "true").lower() == "true"
        self.feedback_retention_days: int = int(os.getenv("FEEDBACK_RETENTION_DAYS", "90"))
        
        # 📋 프로젝트 선택 설정
        self.auto_indexing_enabled: bool = os.getenv("AUTO_INDEXING_ENABLED", "true").lower() == "true"
        self.selective_indexing_enabled: bool = os.getenv("SELECTIVE_INDEXING_ENABLED", "true").lower() == "true"
        self.auto_indexing_interval: int = int(os.getenv("AUTO_INDEXING_INTERVAL", "300"))
        self.project_whitelist_file: str = os.getenv("PROJECT_WHITELIST_FILE", "/data/project_whitelist.json")
        self.host_projects_path: str = os.getenv("HOST_PROJECTS_PATH", "/host_projects")
        self.max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))
        self.max_files_per_project: int = int(os.getenv("MAX_FILES_PER_PROJECT", "1000"))
        self.max_workers: int = int(os.getenv("MAX_WORKERS", "10"))
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
        self.log_file: str = os.getenv("LOG_FILE", "/data/logs/mcp_server.log")
        self.file_watcher_enabled: bool = os.getenv("FILE_WATCHER_ENABLED", "false").lower() == "true"
        self.file_watcher_debounce: float = float(os.getenv("FILE_WATCHER_DEBOUNCE", "2.0"))
        self.vector_search_limit: int = int(os.getenv("VECTOR_SEARCH_LIMIT", "10"))
        self.backup_enabled: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        self.backup_interval: int = int(os.getenv("BACKUP_INTERVAL", "86400"))
        self.backup_retention_days: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

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