"""
성능 최적화 모듈
"""

import asyncio
import gc
import logging
import time
import psutil
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import json
import hashlib
from config import settings

logger = logging.getLogger(__name__)

class LRUCache:
    """LRU 캐시 구현"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key not in self.cache:
            return None
        
        # TTL 확인
        if self._is_expired(key):
            self._remove(key)
            return None
        
        # LRU 순서 업데이트
        self.access_order.remove(key)
        self.access_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """캐시에 값 저장"""
        # 기존 키가 있으면 제거
        if key in self.cache:
            self.access_order.remove(key)
        
        # 용량 초과 시 가장 오래된 항목 제거
        elif len(self.cache) >= self.max_size:
            oldest_key = self.access_order.popleft()
            self._remove(oldest_key)
        
        # 새 항목 추가
        self.cache[key] = value
        self.access_order.append(key)
        self.timestamps[key] = time.time()
    
    def _is_expired(self, key: str) -> bool:
        """TTL 만료 확인"""
        if key not in self.timestamps:
            return True
        
        return (time.time() - self.timestamps[key]) > self.ttl_seconds
    
    def _remove(self, key: str):
        """캐시에서 항목 제거"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear_expired(self) -> int:
        """만료된 항목 정리"""
        expired_keys = []
        current_time = time.time()
        
        for key, timestamp in self.timestamps.items():
            if (current_time - timestamp) > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove(key)
            if key in self.access_order:
                self.access_order.remove(key)
        
        return len(expired_keys)
    
    def clear(self):
        """캐시 전체 초기화"""
        self.cache.clear()
        self.access_order.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """캐시 크기"""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_ratio": getattr(self, "_hit_count", 0) / max(getattr(self, "_total_requests", 1), 1),
            "oldest_entry": min(self.timestamps.values()) if self.timestamps else None
        }

class BatchProcessor:
    """배치 처리기"""
    
    def __init__(self, batch_size: int = 50, timeout_seconds: float = 1.0):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.pending_items = []
        self.last_flush = time.time()
        self.lock = asyncio.Lock()
    
    async def add_item(self, item: Any, processor_func: Callable[[List[Any]], Any]) -> Any:
        """아이템을 배치에 추가하고 필요시 처리"""
        async with self.lock:
            self.pending_items.append((item, processor_func))
            
            # 배치 크기에 도달하거나 타임아웃 시 처리
            if (len(self.pending_items) >= self.batch_size or 
                time.time() - self.last_flush > self.timeout_seconds):
                return await self._flush_batch()
    
    async def _flush_batch(self) -> List[Any]:
        """배치 처리 실행"""
        if not self.pending_items:
            return []
        
        # 함수별로 그룹화
        grouped_items = defaultdict(list)
        for item, func in self.pending_items:
            grouped_items[func].append(item)
        
        results = []
        for func, items in grouped_items.items():
            try:
                batch_result = await func(items)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            except Exception as e:
                logger.error(f"배치 처리 실패: {e}")
                results.extend([None] * len(items))
        
        self.pending_items.clear()
        self.last_flush = time.time()
        
        return results
    
    async def force_flush(self) -> List[Any]:
        """강제 배치 처리"""
        async with self.lock:
            return await self._flush_batch()

class ResourceMonitor:
    """리소스 모니터링"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.request_count = 0
        self.memory_samples = deque(maxlen=100)
        self.cpu_samples = deque(maxlen=100)
    
    def get_memory_usage(self) -> float:
        """메모리 사용량 조회 (MB)"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.memory_samples.append(memory_mb)
        return memory_mb
    
    def get_cpu_usage(self) -> float:
        """CPU 사용량 조회 (%)"""
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        return cpu_percent
    
    def should_trigger_gc(self) -> bool:
        """가비지 컬렉션 트리거 여부"""
        self.request_count += 1
        
        # 요청 수 기준
        if self.request_count % settings.garbage_collection_threshold == 0:
            return True
        
        # 메모리 사용량 기준
        memory_usage = self.get_memory_usage()
        if memory_usage > settings.max_memory_usage_mb * 0.8:  # 80% 임계점
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """리소스 통계"""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count,
            "memory_usage_mb": self.get_memory_usage(),
            "cpu_usage_percent": self.get_cpu_usage(),
            "avg_memory_mb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            "avg_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            "memory_limit_mb": settings.max_memory_usage_mb,
            "cpu_limit_percent": settings.max_cpu_usage_percent
        }

class RateLimiter:
    """API 속도 제한기"""
    
    def __init__(self, rate_per_minute: int = 100, burst_size: int = 10):
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """요청 허용 여부 확인"""
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # 1분 이전 요청들 제거
            while client_requests and client_requests[0] < now - 60:
                client_requests.popleft()
            
            # 버스트 크기 제한
            if len(client_requests) >= self.burst_size:
                return False
            
            # 분당 요청 수 제한
            if len(client_requests) >= self.rate_per_minute:
                return False
            
            # 요청 기록
            client_requests.append(now)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """속도 제한 통계"""
        total_clients = len(self.requests)
        total_requests = sum(len(requests) for requests in self.requests.values())
        
        return {
            "total_clients": total_clients,
            "total_requests_last_minute": total_requests,
            "rate_limit": self.rate_per_minute,
            "burst_size": self.burst_size
        }

class PerformanceOptimizer:
    """성능 최적화 관리자"""
    
    def __init__(self):
        self.cache = LRUCache(
            max_size=settings.max_cache_size,
            ttl_seconds=settings.cache_ttl_seconds
        ) if settings.enable_search_cache else None
        
        self.batch_processor = BatchProcessor(
            batch_size=settings.embedding_batch_size,
            timeout_seconds=1.0
        )
        
        self.resource_monitor = ResourceMonitor()
        self.rate_limiter = RateLimiter(
            rate_per_minute=settings.rate_limit_per_minute,
            burst_size=settings.rate_limit_burst_size
        )
        
        # 성능 통계
        self.operation_times = defaultdict(list)
        self.slow_operations = []
        
        # 백그라운드 정리 태스크 시작
        if self.cache:
            asyncio.create_task(self._cleanup_task())
    
    async def _cleanup_task(self):
        """주기적 정리 작업"""
        while True:
            try:
                # 만료된 캐시 항목 정리
                if self.cache:
                    expired_count = self.cache.clear_expired()
                    if expired_count > 0:
                        logger.info(f"만료된 캐시 항목 {expired_count}개 정리")
                
                # 가비지 컬렉션
                if (hasattr(settings, 'enable_memory_optimization') and 
                    settings.enable_memory_optimization and 
                    self.resource_monitor.should_trigger_gc()):
                    collected = gc.collect()
                    logger.info(f"가비지 컬렉션 완료: {collected}개 객체 해제")
                
                # 30초마다 실행
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"정리 작업 실패: {e}")
                await asyncio.sleep(60)  # 오류 시 더 길게 대기
    
    def get_cache_key(self, operation: str, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = {"op": operation, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_cached_result(self, key: str) -> Optional[Any]:
        """캐시된 결과 조회"""
        if not self.cache:
            return None
        
        result = self.cache.get(key)
        if result is not None:
            logger.debug(f"캐시 히트: {key}")
        
        return result
    
    async def cache_result(self, key: str, result: Any):
        """결과 캐싱"""
        if self.cache:
            self.cache.put(key, result)
            logger.debug(f"캐시 저장: {key}")
    
    async def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """배치 처리"""
        return await self.batch_processor.add_item(items, processor_func)
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """속도 제한 확인"""
        return await self.rate_limiter.is_allowed(client_id)
    
    def record_operation_time(self, operation: str, duration: float):
        """작업 시간 기록"""
        self.operation_times[operation].append(duration)
        
        # 최근 100개만 유지
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation].pop(0)
        
        # 느린 작업 기록
        if duration > settings.slow_query_threshold_seconds:
            self.slow_operations.append({
                "operation": operation,
                "duration": duration,
                "timestamp": time.time()
            })
            
            # 최근 50개만 유지
            if len(self.slow_operations) > 50:
                self.slow_operations.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        stats = {
            "resource_monitor": self.resource_monitor.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "slow_operations": self.slow_operations[-10:],  # 최근 10개만
            "operation_stats": {}
        }
        
        # 캐시 통계
        if self.cache:
            stats["cache"] = self.cache.stats()
        
        # 작업별 통계
        for operation, times in self.operation_times.items():
            if times:
                stats["operation_stats"][operation] = {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times)
                }
        
        return stats

# 전역 성능 최적화 인스턴스
performance_optimizer = PerformanceOptimizer()

def optimize_performance(
    operation_name: str,
    cache_key_generator: Optional[Callable] = None,
    enable_caching: bool = True,
    enable_batch_processing: bool = False
):
    """성능 최적화 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 캐시 확인
            cache_key = None
            if enable_caching and cache_key_generator:
                cache_key = cache_key_generator(*args, **kwargs)
                cached_result = await performance_optimizer.get_cached_result(cache_key)
                if cached_result is not None:
                    return cached_result
            
            try:
                # 함수 실행
                if enable_batch_processing and hasattr(func, '_batch_processor'):
                    result = await performance_optimizer.process_batch(args[0], func)
                else:
                    result = await func(*args, **kwargs)
                
                # 결과 캐싱
                if enable_caching and cache_key:
                    await performance_optimizer.cache_result(cache_key, result)
                
                return result
                
            finally:
                # 성능 기록
                duration = time.time() - start_time
                performance_optimizer.record_operation_time(operation_name, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_optimizer.record_operation_time(operation_name, duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def enable_batch_processing(batch_size: int = 50):
    """배치 처리 활성화 데코레이터"""
    def decorator(func: Callable) -> Callable:
        func._batch_processor = True
        func._batch_size = batch_size
        return func
    
    return decorator 