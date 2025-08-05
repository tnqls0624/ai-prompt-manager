"""
고급 에러 처리 및 로깅 모듈
"""

import logging
import traceback
import functools
import asyncio
from typing import Any, Callable, Dict, Optional, Type, Union
from datetime import datetime
import json
from pathlib import Path
from enum import Enum

class ErrorLevel(Enum):
    """에러 레벨 정의"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """에러 카테고리 정의"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    AI_SERVICE = "ai_service"
    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"

class ErrorHandler:
    """통합 에러 처리 클래스"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 에러 로그 파일 설정
        self.error_log_file = self.log_dir / "errors.log"
        self.performance_log_file = self.log_dir / "performance.log"
        
        # 로거 설정
        self.error_logger = self._setup_logger("error_handler", self.error_log_file)
        self.performance_logger = self._setup_logger("performance", self.performance_log_file)
        
        # 에러 통계
        self.error_stats = {
            "total_errors": 0,
            "by_category": {},
            "by_level": {},
            "recent_errors": []
        }
        
        # 성능 통계
        self.performance_stats = {
            "slow_operations": [],
            "avg_response_time": 0.0,
            "total_operations": 0
        }
    
    def _setup_logger(self, name: str, log_file: Path) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        level: ErrorLevel = ErrorLevel.MEDIUM,
        user_message: Optional[str] = None
    ) -> str:
        """에러 로깅"""
        
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        
        error_info = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "level": level.value,
            "context": context,
            "user_message": user_message,
            "traceback": traceback.format_exc()
        }
        
        # 에러 로그 작성
        self.error_logger.error(json.dumps(error_info, indent=2))
        
        # 통계 업데이트
        self._update_error_stats(error_info)
        
        # 심각한 에러의 경우 알림 (실제 환경에서는 모니터링 시스템 연동)
        if level == ErrorLevel.CRITICAL:
            self._handle_critical_error(error_info)
        
        return error_id
    
    def _update_error_stats(self, error_info: Dict[str, Any]):
        """에러 통계 업데이트"""
        self.error_stats["total_errors"] += 1
        
        # 카테고리별 통계
        category = error_info["category"]
        self.error_stats["by_category"][category] = self.error_stats["by_category"].get(category, 0) + 1
        
        # 레벨별 통계
        level = error_info["level"]
        self.error_stats["by_level"][level] = self.error_stats["by_level"].get(level, 0) + 1
        
        # 최근 에러 (최대 100개)
        self.error_stats["recent_errors"].append({
            "error_id": error_info["error_id"],
            "timestamp": error_info["timestamp"],
            "error_type": error_info["error_type"],
            "category": error_info["category"],
            "level": error_info["level"]
        })
        
        if len(self.error_stats["recent_errors"]) > 100:
            self.error_stats["recent_errors"].pop(0)
    
    def _handle_critical_error(self, error_info: Dict[str, Any]):
        """심각한 에러 처리"""
        critical_log = {
            "CRITICAL_ERROR": True,
            "error_id": error_info["error_id"],
            "timestamp": error_info["timestamp"],
            "error_type": error_info["error_type"],
            "context": error_info["context"]
        }
        
        # 별도의 심각한 에러 로그 파일에 저장
        critical_log_file = self.log_dir / "critical_errors.log"
        with open(critical_log_file, "a") as f:
            f.write(json.dumps(critical_log) + "\n")
        
        # 콘솔에도 출력
        print(f"🚨 CRITICAL ERROR: {error_info['error_id']}")
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        context: Dict[str, Any],
        threshold: float = 1.0
    ):
        """성능 로깅"""
        
        perf_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "context": context,
            "is_slow": duration > threshold
        }
        
        self.performance_logger.info(json.dumps(perf_info))
        
        # 성능 통계 업데이트
        self._update_performance_stats(perf_info)
    
    def _update_performance_stats(self, perf_info: Dict[str, Any]):
        """성능 통계 업데이트"""
        self.performance_stats["total_operations"] += 1
        
        # 평균 응답 시간 계산
        current_avg = self.performance_stats["avg_response_time"]
        total_ops = self.performance_stats["total_operations"]
        new_avg = (current_avg * (total_ops - 1) + perf_info["duration"]) / total_ops
        self.performance_stats["avg_response_time"] = new_avg
        
        # 느린 작업 기록
        if perf_info["is_slow"]:
            self.performance_stats["slow_operations"].append({
                "timestamp": perf_info["timestamp"],
                "operation": perf_info["operation"],
                "duration": perf_info["duration"],
                "context": perf_info["context"]
            })
            
            # 최대 50개의 느린 작업만 유지
            if len(self.performance_stats["slow_operations"]) > 50:
                self.performance_stats["slow_operations"].pop(0)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 조회"""
        return self.error_stats.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        return self.performance_stats.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """시스템 건강 상태 조회"""
        recent_errors = len([
            err for err in self.error_stats["recent_errors"]
            if datetime.fromisoformat(err["timestamp"]) > datetime.now().replace(hour=datetime.now().hour - 1)
        ])
        
        critical_errors = self.error_stats["by_level"].get(ErrorLevel.CRITICAL.value, 0)
        avg_response_time = self.performance_stats["avg_response_time"]
        
        # 건강 상태 결정
        status = "healthy"
        if critical_errors > 0:
            status = "critical"
        elif recent_errors > 10:
            status = "warning"
        elif avg_response_time > 2.0:
            status = "degraded"
        
        return {
            "status": status,
            "recent_errors": recent_errors,
            "critical_errors": critical_errors,
            "avg_response_time": avg_response_time,
            "total_errors": self.error_stats["total_errors"],
            "total_operations": self.performance_stats["total_operations"]
        }

# 전역 에러 핸들러 인스턴스
error_handler = ErrorHandler()

# 데코레이터 함수들

def handle_errors(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    level: ErrorLevel = ErrorLevel.MEDIUM,
    user_message: Optional[str] = None,
    return_on_error: Any = None
):
    """에러 처리 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # 인수 길이 제한
                    "kwargs": str(kwargs)[:200]
                }
                
                error_id = error_handler.log_error(
                    error=e,
                    context=context,
                    category=category,
                    level=level,
                    user_message=user_message
                )
                
                # 에러 발생 시 기본 반환값 또는 재발생
                if return_on_error is not None:
                    return return_on_error
                
                # 사용자 친화적 에러 메시지와 함께 재발생
                if user_message:
                    raise Exception(f"{user_message} (Error ID: {error_id})") from e
                else:
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                
                error_id = error_handler.log_error(
                    error=e,
                    context=context,
                    category=category,
                    level=level,
                    user_message=user_message
                )
                
                if return_on_error is not None:
                    return return_on_error
                
                if user_message:
                    raise Exception(f"{user_message} (Error ID: {error_id})") from e
                else:
                    raise
        
        # 비동기 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def measure_performance(
    operation_name: Optional[str] = None,
    threshold: float = 1.0
):
    """성능 측정 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
                
                error_handler.log_performance(
                    operation=operation_name or func.__name__,
                    duration=duration,
                    context=context,
                    threshold=threshold
                )
                
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "error": str(e)
                }
                
                error_handler.log_performance(
                    operation=operation_name or func.__name__,
                    duration=duration,
                    context=context,
                    threshold=threshold
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
                
                error_handler.log_performance(
                    operation=operation_name or func.__name__,
                    duration=duration,
                    context=context,
                    threshold=threshold
                )
                
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "error": str(e)
                }
                
                error_handler.log_performance(
                    operation=operation_name or func.__name__,
                    duration=duration,
                    context=context,
                    threshold=threshold
                )
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def validate_input(validation_func: Callable[[Any], bool], error_message: str = "Invalid input"):
    """입력 검증 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 첫 번째 인수 (보통 데이터)에 대해 검증
            if args and not validation_func(args[0]):
                raise ValueError(error_message)
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if args and not validation_func(args[0]):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# 공통 검증 함수들
def validate_project_id(project_id: str) -> bool:
    """프로젝트 ID 검증"""
    return isinstance(project_id, str) and len(project_id) > 0 and len(project_id) <= 100

def validate_prompt_content(content: str) -> bool:
    """프롬프트 내용 검증"""
    return isinstance(content, str) and len(content.strip()) > 0 and len(content) <= 10000

def validate_file_path(file_path: str) -> bool:
    """파일 경로 검증"""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False

# 에러 복구 유틸리티
class RetryableError(Exception):
    """재시도 가능한 에러"""
    pass

async def retry_on_failure(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """실패 시 재시도"""
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = delay * (backoff ** attempt)
            await asyncio.sleep(wait_time)
            
            error_handler.log_error(
                error=e,
                context={
                    "function": func.__name__ if hasattr(func, '__name__') else str(func),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "wait_time": wait_time
                },
                category=ErrorCategory.SYSTEM,
                level=ErrorLevel.LOW,
                user_message=f"재시도 중... (시도 {attempt + 1}/{max_retries})"
            )

# 호환성을 위한 별칭 생성
performance_monitor = measure_performance 