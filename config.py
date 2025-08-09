import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 🔐 OpenAI 설정 (환경변수에서만 가져오기)
    openai_api_key: Optional[str] = None
    
    # 🤖 임베딩 모델 설정
    embedding_model_type: str = "deepseek"  # "openai" 또는 "deepseek"
    deepseek_api_base: str = "http://deepseek-r1:11434"  # DeepSeek R1 서버 URL (Ollama 기본 포트)
    deepseek_embedding_model: str = "nomic-embed-text"  # 임베딩 전용 모델
    deepseek_llm_model: str = "r1-1776:latest"  # LLM 전용 모델
    local_embedding_model_path: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
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
    max_concurrent_requests: int = 50  # 10 → 50으로 증가
    request_timeout_seconds: int = 60  # 30 → 60으로 증가
    
    # 임베딩 배치 처리 설정
    embedding_batch_size: int = 100  # 50 → 100으로 증가
    max_concurrent_embeddings: int = 20  # 새로 추가
    
    # 파일 처리 최적화 설정
    max_concurrent_files: int = 100  # 새로 추가
    file_batch_size: int = 200  # 새로 추가
    chunk_overlap_ratio: float = 0.1  # 새로 추가
    
    # 벡터 DB 최적화 설정
    chroma_batch_size: int = 500  # 새로 추가
    enable_parallel_indexing: bool = True  # 새로 추가
    
    # 메모리 최적화 설정
    max_file_size_mb: int = 50  # 새로 추가
    enable_file_compression: bool = True  # 새로 추가
    
    # 데이터베이스 연결 풀링
    db_connection_pool_size: int = 10
    db_connection_timeout: int = 30
    
    # 검색 최적화
    search_result_cache_ttl: int = 300
    similarity_search_algorithm: str = "cosine"
    # 하이브리드 검색 가중치 (튜너블)
    hybrid_semantic_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3
    recency_weight: float = 0.1
    complexity_weight: float = 0.1
    # TF-IDF 인덱스 TTL (초)
    tfidf_index_ttl_seconds: int = 300
    
    # 속도 제한 설정 (새로 추가)
    rate_limit_per_minute: int = 600  # 분당 요청 수
    rate_limit_burst_size: int = 100  # 버스트 크기
    
    # 메모리 관리 설정 (새로 추가)
    enable_memory_optimization: bool = True
    max_memory_usage_mb: int = 2048  # 최대 메모리 사용량 (MB)
    max_cpu_usage_percent: float = 80.0  # 최대 CPU 사용률
    garbage_collection_threshold: int = 100  # GC 트리거 요청 수
    slow_query_threshold_seconds: float = 2.0  # 느린 쿼리 임계값
    
    # 백그라운드 작업 설정
    enable_background_tasks: bool = True
    background_task_interval: int = 60
    
    # 🔄 피드백 시스템 설정
    feedback_learning_rate: float = 0.1  # 학습률 (0.0 ~ 1.0)
    feedback_decay_rate: float = 0.95  # 시간에 따른 피드백 가중치 감소
    min_feedback_count: int = 3  # 최소 피드백 수
    feedback_confidence_threshold: float = 0.7  # 피드백 신뢰도 임계값
    enable_feedback_learning: bool = True  # 피드백 학습 활성화
    feedback_retention_days: int = 90  # 피드백 보존 기간
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 전역 설정 인스턴스
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