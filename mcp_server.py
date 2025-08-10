#!/usr/bin/env python3
"""
FastMCP 기반 MCP 서버
Cursor → Python FastMCP Server (SSE) → LangChain/ChromaDB
"""

from mcp.server.fastmcp import FastMCP
import logging
import sys
import os
from typing import Dict, List, Any, Optional, AsyncIterator, Generator
import asyncio
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import Response
try:
    import jwt as _jwt
except ImportError:
    _jwt = None
from sse_starlette import EventSourceResponse
import difflib
import time
from pathlib import Path
import aiofiles
import hashlib
from datetime import datetime
import aiohttp
import math
from collections import defaultdict, deque
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

# 기존 서비스들 import
from services.vector_service import VectorService
from services.prompt_enhancement_service import PromptEnhancementService
from services.file_indexing_service import FileIndexingService
from services.fast_indexing_service import FastIndexingService
from services.advanced_analytics import AdvancedAnalyticsService
from services.feedback_service import FeedbackService
from services.langchain_rag_service import LangChainRAGService
from services.file_watcher_service import FileWatcherService
from services.auto_indexing_service import AutoIndexingService
from services.error_handler import (
    error_handler,
    handle_errors,
    measure_performance,
    validate_input,
    validate_project_id,
    validate_prompt_content,
    ErrorCategory,
    ErrorLevel
)
from config import settings
from models.prompt_models import PromptHistory, PromptType
import uuid
from logging.handlers import RotatingFileHandler

# 로그 디렉토리 생성 (환경변수 우선, 기본값: /data/logs)
log_dir = os.getenv('LOG_DIR', settings.log_dir)
if log_dir == "/app/logs":  # config.py 기본값인 경우 data/logs로 변경
    log_dir = "/data/logs"
os.makedirs(log_dir, exist_ok=True)

# 로깅 핸들러 설정
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 콘솔 핸들러
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# 파일 핸들러 (로테이션) - 기본값 사용
file_handler = RotatingFileHandler(
    filename=os.path.join(log_dir, "mcp_server.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB 기본값
    backupCount=5,  # 백업 파일 5개 기본값
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# 에러 전용 파일 핸들러
error_file_handler = RotatingFileHandler(
    filename=os.path.join(log_dir, "mcp_server_error.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB 기본값
    backupCount=5,  # 백업 파일 5개 기본값
    encoding='utf-8'
)
error_file_handler.setFormatter(log_formatter)
error_file_handler.setLevel(logging.ERROR)

# 감사 로그 파일 경로 (JSONL)
audit_log_path = os.path.join(log_dir, "audit.jsonl")

def _project_audit_path(project_id: str) -> str:
    safe_id = (project_id or "default").replace("/", "_")[:64]
    return os.path.join(log_dir, f"audit_{safe_id}.jsonl")

async def _rotate_if_needed(path: str, max_bytes: int = 5 * 1024 * 1024, backups: int = 1) -> None:
    try:
        if os.path.exists(path) and os.path.getsize(path) > max_bytes:
            for i in range(backups, 0, -1):
                src = f"{path}.{i}" if i > 0 else path
                dst = f"{path}.{i+1}"
                if os.path.exists(src):
                    try:
                        os.replace(src, dst)
                    except Exception:
                        pass
            try:
                os.replace(path, f"{path}.1")
            except Exception:
                pass
    except Exception:
        pass

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    handlers=[console_handler, file_handler, error_file_handler]
)

logger = logging.getLogger("mcp-server")

# uvicorn access 로그 중 /metrics 요청은 기록하지 않도록 필터 추가
class _SuppressMetricsAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            message = record.getMessage()
        except Exception:
            try:
                message = str(record.msg)
            except Exception:
                message = ""
        return "/metrics" not in message

try:
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(_SuppressMetricsAccessFilter())
except Exception:
    pass

# FastMCP 서버 생성
mcp = FastMCP(
    name="Prompt Enhancement MCP Server",
    debug=True,
    host="0.0.0.0",  # Docker 컨테이너 외부 접근 허용
    port=8000
)

# 🎯 FastMCP의 SSE 앱 활용 (통합된 접근법)
# FastMCP는 내부적으로 Starlette를 사용하므로 custom_route로 엔드포인트 추가

# 전역 서비스 인스턴스
vector_service: Optional[VectorService] = None
enhancement_service: Optional[PromptEnhancementService] = None
file_indexing_service: Optional[FileIndexingService] = None
fast_indexing_service: Optional[FastIndexingService] = None
analytics_service: Optional[AdvancedAnalyticsService] = None
feedback_service: Optional[FeedbackService] = None
langchain_rag_service: Optional[LangChainRAGService] = None
file_watcher_service: Optional[FileWatcherService] = None
auto_indexing_service: Optional[AutoIndexingService] = None

# 초기화 상태 관리
_initialization_complete = asyncio.Event()
_services_initialized = False

# SSE 연결 관리
active_connections: Dict[str, asyncio.Queue] = {}

# 간단한 레이트 리미터 (분당 요청 수 + 버스트 제한)
_rate_limit_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
_project_quota_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

def _get_client_key(request: Request) -> str:
    try:
        # 프록시 고려
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client_host = request.client.host if request and request.client else "unknown"
        return client_host
    except Exception:
        return "unknown"

def _check_rate_limit(request: Request) -> bool:
    now = time.time()
    key = _get_client_key(request)
    bucket = _rate_limit_buckets[key]
    # 60초 이전 항목 제거
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    # 설정 값
    max_per_min = getattr(settings, "rate_limit_per_minute", 600)
    burst = getattr(settings, "rate_limit_burst_size", 100)
    if len(bucket) >= max_per_min or (burst and len(bucket) >= burst):
        return False
    bucket.append(now)
    return True

def _check_project_quota(project_id: str) -> bool:
    quota = getattr(settings, 'project_quota_per_minute', 0) or 0
    if quota <= 0:
        return True
    now = time.time()
    bucket = _project_quota_buckets[project_id]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    if len(bucket) >= quota:
        return False
    bucket.append(now)
    return True

# ---------------------------------------------------------------------------
# Metrics (Prometheus optional)
# ---------------------------------------------------------------------------
if _PROM_AVAILABLE:
    REQUEST_COUNT = Counter('mcp_requests_total', 'Total requests', ['endpoint'])
    REQUEST_LATENCY = Histogram('mcp_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
    REQUEST_ERRORS = Counter('mcp_request_errors_total', 'Total server errors', ['endpoint'])
else:
    REQUEST_COUNT = None
    REQUEST_LATENCY = None
    REQUEST_ERRORS = None

class _RequestTimer:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._ctx = None
    def __enter__(self):
        if REQUEST_LATENCY:
            self._ctx = REQUEST_LATENCY.labels(self.endpoint).time()
            self._ctx.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._ctx:
            self._ctx.__exit__(exc_type, exc, tb)

# ---------------------------------------------------------------------------
# RAG: Generate minimal edits (code edit endpoint)
# ---------------------------------------------------------------------------
@mcp.custom_route(path="/api/v1/rag/generate-edit", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="RAG 기반 코드 에디트 생성 중 오류가 발생했습니다."
)
@measure_performance(operation_name="rag_generate_edit", threshold=10.0)
async def rag_generate_edit(request):
    """RAG 기반 코드 에디트(JSON edits) 생성 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request) or not _check_jwt(request):
            return JSONResponse({"error": "Unauthorized", "success": False}, status_code=401)
        if REQUEST_COUNT:
            REQUEST_COUNT.labels('/api/v1/rag/generate-edit').inc()

        data = await request.json()
        instruction = data.get("instruction", "").strip()
        project_id = data.get("project_id", "default")
        file_path = data.get("file_path")
        diff_context = data.get("diff_context", "")
        context_limit = int(data.get("context_limit", 5))
        include_snippet = bool(data.get("include_snippet", True))

        if not instruction:
            return JSONResponse({"error": "instruction is required", "success": False}, status_code=400)

        # 컨텍스트 수집
        with _RequestTimer('/api/v1/rag/generate-edit'):
            enhanced = await langchain_rag_service.generate_enhanced_prompt(
                user_prompt=instruction,
                project_id=project_id,
                context_limit=context_limit
            )
            head_snippet = None
            if file_path and include_snippet:
                head_snippet = await _read_head_snippet(file_path, max_lines=200)

            # LLM 사용 시 JSON edits 생성 시도
            try:
                if getattr(langchain_rag_service, 'llm', None):
                    import json as _json
                    prompt = (
                        "You are a code refactoring assistant. Generate a minimal set of edits in strict JSON.\n"
                        "Schema: {\"edits\":[{\"file_path\":string,\"start_line\":number,\"end_line\":number,\"new_text\":string}],\"explanations\":[string]}\n"
                        "Only output the JSON. No prose.\n\n"
                        f"Instruction: {instruction}\n"
                        f"TargetFile: {file_path or 'unspecified'}\n"
                        f"DiffContext:\n{diff_context}\n\n"
                        f"ProjectContext:\n{enhanced.get('context','')}\n"
                        f"HeadSnippet:\n{(head_snippet or {}).get('content','')}\n"
                    )
                    raw = await langchain_rag_service.llm.arun(prompt=prompt)
                    from models.enhanced_models import CodeEditsResponse, CodeEdit
                    # JSON 파싱 및 스키마 검증
                    try:
                        payload = _json.loads(raw)
                    except Exception:
                        cleaned = raw.strip().strip('`')
                        payload = _json.loads(cleaned)
                    edits_raw = payload.get('edits', [])
                    explanations = payload.get('explanations', [])
                    edits: List[CodeEdit] = []
                    for e in edits_raw:
                        try:
                            edit = CodeEdit(**e).normalize()
                            edits.append(edit)
                        except Exception:
                            continue
                    validated = CodeEditsResponse(edits=edits, explanations=explanations)
                    # optional dry-run diff
                    dry_run = bool(data.get('dry_run', False))
                    diff_preview = None
                    if dry_run and file_path and validated.edits:
                        try:
                            # read original file
                            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as fsrc:
                                original = await fsrc.readlines()
                            # apply first edit preview only (for simplicity)
                            e0 = validated.edits[0]
                            start = max(1, e0.start_line) - 1
                            end = max(start, e0.end_line)  # inclusive
                            new_text_lines = (e0.new_text or '').splitlines(keepends=True)
                            preview = original[:start] + new_text_lines + original[end:]
                            diff = difflib.unified_diff(original, preview, fromfile='before', tofile='after')
                            diff_preview = ''.join(diff)[:10000]
                        except Exception:
                            diff_preview = None
                    return JSONResponse({
                        "success": True,
                        "edits": [e.dict() for e in validated.edits],
                        "explanations": validated.explanations,
                        "diff_preview": diff_preview
                    })
            except Exception as e:
                logger.warning(f"generate-edit LLM JSON parsing failed: {e}")

            # 폴백: 템플릿 기반 최소 에디트
            default_edit = {
                "file_path": file_path or "<select a file>",
                "start_line": 1,
                "end_line": 1,
                "new_text": "# TODO: apply minimal change for instruction: " + instruction
            }
            return JSONResponse({
                "success": True,
                "edits": [default_edit],
                "explanations": [
                    "LLM unavailable or failed to produce strict JSON. Returning a minimal edit template."
                ]
            })

    except Exception as e:
        logger.error(f"RAG 기반 코드 에디트 생성 중 오류: {str(e)}")
        return JSONResponse({"error": str(e), "success": False}, status_code=500)


# 간단한 API Key 인증 검사 (옵션)
def _check_api_key(request: Request) -> bool:
    if not getattr(settings, 'require_api_key', False):
        return True
    provided = request.headers.get('x-api-key') or request.query_params.get('api_key')
    expected = getattr(settings, 'api_key', None)
    return bool(expected) and provided == expected

def _check_jwt(request: Request) -> bool:
    if not getattr(settings, 'jwt_enabled', False):
        return True
    if _jwt is None:
        return False
    auth = request.headers.get('authorization') or ''
    if not auth.lower().startswith('bearer '):
        return False
    token = auth.split(' ', 1)[1].strip()
    try:
        _jwt.decode(token, getattr(settings, 'jwt_secret', ''), algorithms=[getattr(settings, 'jwt_algorithms', 'HS256')])
        return True
    except Exception:
        return False

def _audit(event: str, meta: Dict[str, Any]):
    if not getattr(settings, 'audit_log_enabled', False):
        return
    try:
        try:
            if isinstance(event, str) and event.lower().strip() in {"/metrics", "metrics"}:
                return
        except Exception:
            pass
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "meta": meta,
        }
        # 파일에 JSONL로 비동기 기록
        async def _write():
            try:
                # 프로젝트별 파일에도 기록
                proj_path = _project_audit_path(str(meta.get('project_id', 'default')))
                await _rotate_if_needed(audit_log_path)
                await _rotate_if_needed(proj_path)
                async with aiofiles.open(audit_log_path, mode='a', encoding='utf-8') as f:
                    await f.write(json.dumps(record, ensure_ascii=False) + "\n")
                async with aiofiles.open(proj_path, mode='a', encoding='utf-8') as f2:
                    await f2.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"감사 로그 기록 실패: {e}")
        # 백그라운드 태스크로 실행
        asyncio.create_task(_write())
    except Exception:
        pass

class SSEEventType:
    """SSE 이벤트 타입"""
    ENHANCEMENT_START = "enhancement_start"
    ENHANCEMENT_PROGRESS = "enhancement_progress"
    ENHANCEMENT_COMPLETE = "enhancement_complete"
    CONTEXT_SEARCH = "context_search"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

# ---------------------------------------------------------------------------
# Utility: File snippet reading
# ---------------------------------------------------------------------------
async def _read_file_snippet(file_path: str, start_line: int, end_line: int, max_lines: int = 500) -> Dict[str, Any]:
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if end_line - start_line + 1 > max_lines:
        end_line = start_line + max_lines - 1
    try:
        result_lines: List[str] = []
        current = 1
        async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
            async for line in f:
                if current > end_line:
                    break
                if start_line <= current <= end_line:
                    result_lines.append(line)
                current += 1
        return {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "content": ''.join(result_lines)
        }
    except Exception as e:
        return {
            "error": str(e)
        }

async def _read_head_snippet(file_path: str, max_lines: int = 300) -> Dict[str, Any]:
    """파일의 앞부분 일부를 반환 (컨텍스트가 없을 때 자동 프롬프트용)"""
    return await _read_file_snippet(file_path, 1, max_lines, max_lines=max_lines)

def _guess_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".json": "json",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
    }.get(ext, "plaintext")

@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.CRITICAL,
    user_message="서비스 초기화에 실패했습니다."
)
@measure_performance(operation_name="service_initialization", threshold=10.0)
async def initialize_services():
    """모든 서비스 초기화"""
    global vector_service, enhancement_service, file_indexing_service, fast_indexing_service, analytics_service, feedback_service, langchain_rag_service, file_watcher_service, auto_indexing_service, _services_initialized
    
    try:
        # 설정 검증
        logger.info("환경 설정 검증 중...")
        required_settings = [
            'chroma_db_path', 'embedding_model_type', 
            'max_concurrent_requests', 'embedding_batch_size'
        ]
        
        for setting in required_settings:
            if not hasattr(settings, setting):
                logger.warning(f"설정 누락: {setting}, 기본값 사용")
        
        # 벡터 서비스 초기화
        logger.info("벡터 서비스 초기화 중...")
        vector_service = VectorService()
        # ChromaDB는 VectorService 생성자에서 자동 초기화됨
        
        # 프롬프트 향상 서비스
        logger.info("프롬프트 향상 서비스 초기화 중...")
        prompt_service = PromptEnhancementService(vector_service)
        
        # 파일 인덱싱 서비스
        logger.info("파일 인덱싱 서비스 초기화 중...")
        file_indexing_service = FileIndexingService(vector_service)
        
        # 고속 인덱싱 서비스
        logger.info("고속 인덱싱 서비스 초기화 중...")
        fast_indexing_service = FastIndexingService(vector_service)
        
        # 고급 분석 서비스
        logger.info("고급 분석 서비스 초기화 중...")
        analytics_service = AdvancedAnalyticsService()  # 인자 없이 생성
        
        # 피드백 서비스
        logger.info("피드백 서비스 초기화 중...")
        feedback_service = FeedbackService(vector_service)  # vector_service 필요
        
        # LangChain RAG 서비스
        logger.info("LangChain RAG 서비스 초기화 중...")
        langchain_rag_service = LangChainRAGService(vector_service)
        
        # 파일 감시 서비스
        logger.info("파일 감시 서비스 초기화 중...")
        file_watcher_service = FileWatcherService(vector_service)
        
        # 자동 인덱싱 서비스
        logger.info("자동 인덱싱 서비스 초기화 중...")
        auto_indexing_service = AutoIndexingService(vector_service, file_indexing_service)  # 두 개의 인자 필요
        
        # 전역 변수 설정
        enhancement_service = prompt_service  # 호환성을 위한 별칭
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}", exc_info=True)
        raise
    
    logger.info("모든 서비스가 초기화되었습니다")
    
    # 자동 인덱싱 서비스 시작 (백그라운드에서)
    logger.info("자동 백그라운드 인덱싱 서비스 시작...")
    try:
        await auto_indexing_service.start()
        logger.info("자동 인덱싱 서비스가 성공적으로 시작되었습니다")
    except Exception as e:
        logger.error(f"자동 인덱싱 서비스 시작 중 오류: {e}")

    # 선택적: 서버 시작 시 인덱스 워밍업
    try:
        if getattr(settings, 'warmup_on_start', False):
            raw_ids = getattr(settings, 'warmup_project_ids', None) or ''
            project_ids = [pid.strip() for pid in raw_ids.split(',') if pid.strip()]
            if project_ids:
                logger.info(f"인덱스 워밍업 시작: {project_ids}")
                await asyncio.gather(*[
                    vector_service.warmup_project_indices(pid) for pid in project_ids
                ])
                logger.info("인덱스 워밍업 완료")
    except Exception as e:
        logger.warning(f"워밍업 중 오류: {e}")
    
    # 초기화 완료 표시
    _services_initialized = True
    _initialization_complete.set()
    
    logger.info("모든 서비스가 초기화되었습니다")

@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.HIGH,
    user_message="서비스 초기화 대기 중 오류가 발생했습니다."
)
async def ensure_services_initialized():
    """서비스가 초기화될 때까지 대기"""
    if not _services_initialized:
        logger.info("서비스 초기화 대기 중...")
        await initialize_services()
    
    # 초기화 완료까지 대기 (최대 30초)
    try:
        await asyncio.wait_for(_initialization_complete.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("서비스 초기화 타임아웃")
        raise RuntimeError("서비스 초기화가 30초 내에 완료되지 않았습니다")

async def store_prompt(
    content: str,
    project_id: str,
    prompt_type: str
) -> Dict[str, Any]:
    """
    프롬프트를 벡터 데이터베이스에 저장합니다.
    
    Args:
        content: 저장할 프롬프트 내용
        project_id: 프로젝트 ID
        prompt_type: 프롬프트 타입 ("user_query", "ai_response", "system_prompt", "enhanced_prompt")
    
    Returns:
        저장 결과 정보
    """
    try:
        await ensure_services_initialized()
        
        # PromptHistory 객체 생성
        prompt_history = PromptHistory(
            id=str(uuid.uuid4()),
            project_id=project_id,
            content=content,
            prompt_type=PromptType(prompt_type),
            metadata={}
        )
        
        # 벡터 서비스에 저장
        success = await vector_service.store_prompt_history(prompt_history)
        
        if success:
            return {
                "success": True,
                "id": prompt_history.id,
                "message": "프롬프트가 성공적으로 저장되었습니다"
            }
        else:
            return {
                "success": False,
                "error": "벡터 서비스 저장 실패"
            }
            
    except Exception as e:
        logger.error(f"프롬프트 저장 중 오류: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """SSE 이벤트 생성"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

async def broadcast_to_connection(connection_id: str, event_type: str, data: Dict[str, Any]):
    """특정 연결에 이벤트 브로드캐스트"""
    if connection_id in active_connections:
        event_data = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        await active_connections[connection_id].put(event_data)

# SSE 엔드포인트
@mcp.custom_route(path="/api/v1/sse/{connection_id}", methods=["GET"])
@handle_errors(
    category=ErrorCategory.NETWORK,
    level=ErrorLevel.MEDIUM,
    user_message="SSE 연결 중 오류가 발생했습니다."
)
async def sse_endpoint(request):
    """SSE 연결 엔드포인트"""
    connection_id = request.path_params.get("connection_id")
    logger.info(f"SSE 연결 시작: {connection_id}")
    
    # 연결 큐 생성
    event_queue = asyncio.Queue()
    active_connections[connection_id] = event_queue
    
    async def event_generator():
        try:
            # 연결 확인 메시지
            yield await create_sse_event("connected", {
                "connection_id": connection_id,
                "server_status": "ready"
            })
            
            while True:
                try:
                    # 이벤트 대기 (타임아웃 30초)
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    yield await create_sse_event(event["type"], event["data"])
                except asyncio.TimeoutError:
                    # 하트비트 전송
                    yield await create_sse_event("heartbeat", {
                        "timestamp": time.time()
                    })
                    
        except Exception as e:
            logger.error(f"SSE 이벤트 생성 오류: {e}")
            yield await create_sse_event("error", {
                "error": str(e)
            })
        finally:
            # 연결 정리
            if connection_id in active_connections:
                del active_connections[connection_id]
            logger.info(f"SSE 연결 종료: {connection_id}")
    
    return EventSourceResponse(event_generator())

# 스트리밍 프롬프트 개선 엔드포인트
@mcp.custom_route(path="/api/v1/enhance-prompt-stream/{connection_id}", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프롬프트 개선 중 오류가 발생했습니다."
)
@measure_performance(operation_name="enhance_prompt_stream", threshold=5.0)
async def enhance_prompt_stream(request):
    """스트리밍 프롬프트 개선"""
    await ensure_services_initialized()
    
    # Path parameter 추출
    connection_id = request.path_params.get("connection_id")
    
    # 요청 데이터 파싱
    request_data = await request.json()
    prompt = request_data.get("prompt", "")
    project_id = request_data.get("project_id", "default")
    context_limit = request_data.get("context_limit", 5)
    
    # 입력 검증
    if not validate_prompt_content(prompt):
        raise HTTPException(status_code=400, detail="유효하지 않은 프롬프트 내용입니다.")
    
    if not validate_project_id(project_id):
        raise HTTPException(status_code=400, detail="유효하지 않은 프로젝트 ID입니다.")
    
    # 개선 시작 알림
    await broadcast_to_connection(connection_id, SSEEventType.ENHANCEMENT_START, {
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "project_id": project_id
    })
    
    # PromptEnhanceRequest 객체 생성
    from models.prompt_models import PromptEnhanceRequest
    enhance_request = PromptEnhanceRequest(
        original_prompt=prompt,
        project_id=project_id,
        context_limit=context_limit
    )
    
    # 컨텍스트 검색 알림
    await broadcast_to_connection(connection_id, SSEEventType.CONTEXT_SEARCH, {
        "status": "searching",
        "message": "관련 컨텍스트를 검색 중입니다..."
    })
    
    # 프롬프트 개선 수행
    result = await enhancement_service.enhance_prompt(enhance_request)
    
    # 개선 완료 알림
    await broadcast_to_connection(connection_id, SSEEventType.ENHANCEMENT_COMPLETE, {
        "enhanced_prompt": result.enhanced_prompt,
        "confidence_score": result.confidence_score,
        "context_used": result.context_used,
        "suggestions": result.suggestions
    })
    
    return {
        "success": True,
        "message": "프롬프트 개선이 스트리밍으로 완료되었습니다",
        "connection_id": connection_id
    }

# 스트리밍 프로젝트 인덱싱 엔드포인트
@mcp.custom_route(path="/api/v1/index-project-stream/{connection_id}", methods=["POST"])
@handle_errors(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.MEDIUM,
    user_message="프로젝트 인덱싱 중 오류가 발생했습니다."
)
@measure_performance(operation_name="index_project_stream", threshold=30.0)
async def index_project_stream(request):
    """스트리밍 프로젝트 인덱싱"""
    await ensure_services_initialized()
    
    # Path parameter 추출
    connection_id = request.path_params.get("connection_id")
    
    # 요청 데이터 파싱
    request_data = await request.json()
    project_path = request_data.get("project_path", "")
    project_id = request_data.get("project_id", "default")
    
    # 입력 검증
    if not validate_project_id(project_id):
        raise HTTPException(status_code=400, detail="유효하지 않은 프로젝트 ID입니다.")
    
    if not os.path.exists(project_path):
        raise HTTPException(status_code=400, detail="프로젝트 경로가 존재하지 않습니다.")
    
    # 인덱싱 시작 알림
    await broadcast_to_connection(connection_id, "indexing_start", {
        "project_path": project_path,
        "project_id": project_id
    })
    
    # 진행 상황 모니터링을 위한 콜백 함수
    async def progress_callback(current: int, total: int, message: str):
        await broadcast_to_connection(connection_id, "indexing_progress", {
            "current": current,
            "total": total,
            "message": message,
            "progress": (current / total) * 100 if total > 0 else 0
        })
    
    # 통합 인덱싱 수행 (fast_indexing_service 사용)
    result = await fast_indexing_service.index_project(
        project_path=project_path,
        project_id=project_id,
        progress_callback=progress_callback
    )
    
    # 인덱싱 완료 알림
    await broadcast_to_connection(connection_id, "indexing_complete", {
        "result": result
    })
    
    return {
        "success": True,
        "message": "프로젝트 인덱싱이 스트리밍으로 완료되었습니다",
        "connection_id": connection_id
    }

# 🚀 새로운 파일 업로드 및 병렬 인덱싱 API
@mcp.custom_route(path="/api/v1/upload-files", methods=["POST"])
@handle_errors(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.MEDIUM,
    user_message="파일 업로드 및 인덱싱 중 오류가 발생했습니다."
)
@measure_performance(operation_name="upload_files", threshold=30.0)
async def upload_files(request):
    """파일 업로드 및 병렬 인덱싱"""
    await ensure_services_initialized()
    
    try:
        # 폼 데이터 파싱
        form_data = await request.form()
        project_id = form_data.get("project_id", "default")
        project_name = form_data.get("project_name", "uploaded-project")
        
        # 입력 검증
        if not validate_project_id(project_id):
            return JSONResponse(
                status_code=400,
                content={"error": "유효하지 않은 프로젝트 ID입니다.", "success": False}
            )
        
        # 업로드된 파일들 처리
        uploaded_files = []
        file_count = 0
        
        for key, file in form_data.items():
            if key.startswith("file_") and hasattr(file, 'filename'):
                if file.filename and file.filename.strip():
                    file_content = await file.read()
                    if len(file_content) > 0:
                        uploaded_files.append({
                            "filename": file.filename,
                            "content": file_content.decode('utf-8', errors='ignore'),
                            "size": len(file_content)
                        })
                        file_count += 1
        
        if not uploaded_files:
            return JSONResponse(
                status_code=400,
                content={"error": "업로드된 파일이 없습니다.", "success": False}
            )
        
        logger.info(f"📤 {file_count}개 파일 업로드 완료, 병렬 인덱싱 시작...")
        
        # 병렬 인덱싱 처리
        indexed_files = []
        failed_files = []
        
        # Semaphore로 동시성 제어 (최대 10개 파일 동시 처리)
        semaphore = asyncio.Semaphore(10)
        
        async def process_uploaded_file(file_data):
            async with semaphore:
                try:
                    # 파일 청크를 벡터 DB에 저장
                    await _store_uploaded_file_to_vector_db(
                        file_data["filename"],
                        file_data["content"],
                        project_id
                    )
                    return file_data["filename"]
                except Exception as e:
                    logger.warning(f"파일 인덱싱 실패 {file_data['filename']}: {e}")
                    return None
        
        # 모든 파일을 병렬로 처리
        results = await asyncio.gather(
            *[process_uploaded_file(file_data) for file_data in uploaded_files],
            return_exceptions=True
        )
        
        # 결과 분류
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_files.append(uploaded_files[i]["filename"])
            elif result is not None:
                indexed_files.append(result)
            else:
                failed_files.append(uploaded_files[i]["filename"])
        
        # 프로젝트 컨텍스트도 저장
        from models.prompt_models import ProjectContext
        project_context = ProjectContext(
            project_id=project_id,
            project_name=project_name,
            description=f"업로드된 프로젝트 ({len(indexed_files)}개 파일)",
            tech_stack=_detect_tech_stack_from_files([f["filename"] for f in uploaded_files]),
            file_patterns=list(set([f"*{Path(f['filename']).suffix}" for f in uploaded_files]))
        )
        await vector_service.store_project_context(project_context)
        
        return JSONResponse({
            "success": True,
            "project_id": project_id,
            "project_name": project_name,
            "total_files_uploaded": len(uploaded_files),
            "indexed_files_count": len(indexed_files),
            "failed_files_count": len(failed_files),
            "indexed_files": indexed_files,
            "failed_files": failed_files[:10] if failed_files else [],
            "tech_stack": project_context.tech_stack
        })
        
    except Exception as e:
        logger.error(f"파일 업로드 처리 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"파일 업로드 처리 실패: {str(e)}", "success": False}
        )

# 배치 파일 업로드 API (JSON 방식)
@mcp.custom_route(path="/api/v1/upload-batch", methods=["POST"])
@handle_errors(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.MEDIUM,
    user_message="배치 파일 업로드 중 오류가 발생했습니다."
)
@measure_performance(operation_name="upload_batch", threshold=60.0)
async def upload_batch(request):
    """배치 파일 업로드 및 인덱싱 (JSON 방식)"""
    await ensure_services_initialized()
    
    try:
        # JSON 데이터 파싱
        request_data = await request.json()
        project_id = request_data.get("project_id", "default")
        project_name = request_data.get("project_name", "batch-project")
        files_data = request_data.get("files", [])
        
        # 입력 검증
        if not validate_project_id(project_id):
            return JSONResponse(
                status_code=400,
                content={"error": "유효하지 않은 프로젝트 ID입니다.", "success": False}
            )
        
        if not files_data:
            return JSONResponse(
                status_code=400,
                content={"error": "업로드할 파일 데이터가 없습니다.", "success": False}
            )
        
        logger.info(f"📤 {len(files_data)}개 파일 배치 업로드 시작...")
        
        # 병렬 인덱싱 처리 (더 큰 배치 크기)
        indexed_files = []
        failed_files = []
        
        # 배치 단위로 처리 (200개씩) - 기존 100개에서 증가
        batch_size = 200
        semaphore = asyncio.Semaphore(100)  # 50 → 100으로 증가 (더 높은 동시성)
        
        async def process_file_batch(file_data):
            async with semaphore:
                try:
                    filename = file_data.get("path", file_data.get("filename", "unknown"))
                    content = file_data.get("content", "")
                    
                    if len(content.strip()) < 10:  # 너무 작은 파일 제외
                        return None
                    
                    await _store_uploaded_file_to_vector_db(filename, content, project_id)
                    return filename
                except Exception as e:
                    logger.warning(f"파일 인덱싱 실패 {filename}: {e}")
                    return None
        
        # 배치별로 병렬 처리
        for i in range(0, len(files_data), batch_size):
            batch = files_data[i:i + batch_size]
            logger.info(f"🔄 배치 {i//batch_size + 1}/{(len(files_data) + batch_size - 1)//batch_size} 처리 중...")
            
            batch_results = await asyncio.gather(
                *[process_file_batch(file_data) for file_data in batch],
                return_exceptions=True
            )
            
            # 결과 분류
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_files.append(batch[j].get("path", "unknown"))
                elif result is not None:
                    indexed_files.append(result)
                else:
                    failed_files.append(batch[j].get("path", "unknown"))
        
        # 프로젝트 컨텍스트 저장
        from models.prompt_models import ProjectContext
        project_context = ProjectContext(
            project_id=project_id,
            project_name=project_name,
            description=f"배치 업로드된 프로젝트 ({len(indexed_files)}개 파일)",
            tech_stack=_detect_tech_stack_from_files([f.get("path", "") for f in files_data]),
            file_patterns=list(set([f"*{Path(f.get('path', '')).suffix}" for f in files_data if f.get('path')]))
        )
        await vector_service.store_project_context(project_context)
        
        return JSONResponse({
            "success": True,
            "project_id": project_id,
            "project_name": project_name,
            "total_files_received": len(files_data),
            "indexed_files_count": len(indexed_files),
            "failed_files_count": len(failed_files),
            "success_rate": round((len(indexed_files) / len(files_data)) * 100, 1),
            "tech_stack": project_context.tech_stack,
            "file_patterns": project_context.file_patterns
        })
        
    except Exception as e:
        logger.error(f"배치 업로드 처리 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"배치 업로드 처리 실패: {str(e)}", "success": False}
        )

# 헬퍼 함수들
async def _store_uploaded_file_to_vector_db(filename: str, content: str, project_id: str):
    """업로드된 파일을 벡터 DB에 저장"""
    try:
        # 파일 경로 정보 생성
        file_path = Path(filename)
        
        # 메타데이터 구성
        metadata = {
            "file_path": filename,
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "chunk_index": 0,
            "file_type": "code" if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.jsx', '.tsx'] else "documentation",
            "is_file_content": True,
            "upload_method": "network"
        }
        
        # 파일이 크면 청킹
        if len(content) > 16000:  # 16KB 이상이면 청킹
            chunks = _chunk_text_content(content)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                
                # 고유 ID 생성
                chunk_id = f"{project_id}_upload_{hashlib.md5(f'{filename}_{i}'.encode()).hexdigest()}"
                
                # PromptHistory 객체 생성
                from models.prompt_models import PromptHistory, PromptType
                prompt_history = PromptHistory(
                    id=chunk_id,
                    project_id=project_id,
                    content=chunk,
                    prompt_type=PromptType.SYSTEM_PROMPT,
                    metadata=chunk_metadata,
                    created_at=datetime.now()
                )
                
                await vector_service.store_prompt_history(prompt_history)
        else:
            # 작은 파일은 통째로 저장
            file_id = f"{project_id}_upload_{hashlib.md5(filename.encode()).hexdigest()}"
            
            from models.prompt_models import PromptHistory, PromptType
            prompt_history = PromptHistory(
                id=file_id,
                project_id=project_id,
                content=content,
                prompt_type=PromptType.SYSTEM_PROMPT,
                metadata=metadata,
                created_at=datetime.now()
            )
            
            await vector_service.store_prompt_history(prompt_history)
            
    except Exception as e:
        logger.error(f"파일 벡터 저장 실패 {filename}: {e}")
        raise

def _chunk_text_content(content: str) -> List[str]:
    """텍스트 내용을 청킹"""
    # 기본 청킹 (줄 단위)
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        current_chunk.append(line)
        current_size += len(line)
        
        if current_size > 8000:  # 8KB씩 청킹
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def _detect_tech_stack_from_files(filenames: List[str]) -> List[str]:
    """파일명들로부터 기술 스택 감지"""
    tech_stack = set()
    
    for filename in filenames:
        ext = Path(filename).suffix.lower()
        
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
    
    return sorted(list(tech_stack))

# 시스템 건강 상태 체크 엔드포인트
@mcp.custom_route(path="/api/v1/health", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"status": "error", "message": "건강 상태 확인 중 오류 발생"}
)
async def health_check(request):
    """시스템 건강 상태 확인"""
    health_status = error_handler.get_health_status()
    
    # 서비스 상태 확인
    services_status = {
        "vector_service": vector_service is not None,
        "enhancement_service": enhancement_service is not None,
        "file_indexing_service": file_indexing_service is not None,
        "fast_indexing_service": fast_indexing_service is not None,
        "analytics_service": analytics_service is not None
    }
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "services": services_status,
        "health_stats": health_status,
        "active_connections": len(active_connections)
    })

# 에러 통계 엔드포인트
@mcp.custom_route(path="/api/v1/error-stats", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "에러 통계 조회 실패"}
)
async def get_error_stats(request):
    """에러 통계 조회"""
    return JSONResponse({
        "error_stats": error_handler.get_error_stats(),
        "performance_stats": error_handler.get_performance_stats()
    })

# MCP 도구들 정의

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프롬프트 개선 중 오류가 발생했습니다."
)
@measure_performance(operation_name="enhance_prompt", threshold=3.0)
async def enhance_prompt(
    prompt: str,
    project_id: str = "default",
    context_limit: int = 5
) -> Dict[str, Any]:
    """
    AI 프롬프트를 분석하고 개선 제안을 제공합니다.
    
    Args:
        prompt: 개선할 프롬프트 텍스트
        project_id: 프로젝트 식별자 (기본값: "default")
        context_limit: 컨텍스트 제한 개수 (기본값: 5)
    
    Returns:
        개선된 프롬프트와 제안사항
    """
    logger.info(f"프롬프트 개선 요청: {prompt[:50]}...")
    
    # 입력 검증
    if not validate_prompt_content(prompt):
        return {
            "success": False,
            "error": "유효하지 않은 프롬프트 내용입니다."
        }
    
    if not validate_project_id(project_id):
        return {
            "success": False,
            "error": "유효하지 않은 프로젝트 ID입니다."
        }
    
    await ensure_services_initialized()
    
    # PromptEnhanceRequest 객체 생성
    from models.prompt_models import PromptEnhanceRequest
    request = PromptEnhanceRequest(
        original_prompt=prompt,
        project_id=project_id,
        context_limit=context_limit
    )
    
    result = await enhancement_service.enhance_prompt(request)
    
    logger.info("프롬프트 개선 완료")
    return {
        "success": True,
        "enhanced_prompt": result.enhanced_prompt,
        "suggestions": result.suggestions,
        "context_used": result.context_used,
        "improvement_score": result.confidence_score
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.MEDIUM,
    user_message="대화 저장 중 오류가 발생했습니다."
)
@measure_performance(operation_name="store_conversation", threshold=5.0)
async def store_conversation(
    user_prompt: str,
    ai_response: str,
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    사용자와 AI의 대화를 학습 데이터로 저장합니다.
    
    Args:
        user_prompt: 사용자 프롬프트
        ai_response: AI 응답
        project_id: 프로젝트 식별자 (기본값: "default")
    
    Returns:
        저장 결과 정보
    """
    logger.info(f"대화 저장 요청: 사용자={user_prompt[:30]}..., AI={ai_response[:30]}...")
    
    await ensure_services_initialized()
    
    # 사용자 프롬프트 저장
    user_result = await store_prompt(user_prompt, project_id, "user_query")
    
    # AI 응답 저장  
    ai_result = await store_prompt(ai_response, project_id, "ai_response")
    
    success = user_result.get("success", False) and ai_result.get("success", False)
    
    logger.info("대화 저장 완료" if success else "대화 저장 실패")
    
    return {
        "success": success,
        "message": "대화가 학습 데이터로 저장되었습니다" if success else "저장 중 오류 발생",
        "user_prompt_id": user_result.get("id"),
        "ai_response_id": ai_result.get("id")
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="유사 대화 검색 중 오류가 발생했습니다."
)
@measure_performance(operation_name="search_similar_conversations", threshold=3.0)
async def search_similar_conversations(
    query: str,
    project_id: str = "default",
    limit: int = 5
) -> Dict[str, Any]:
    """
    유사한 대화나 프롬프트를 검색합니다.
    
    Args:
        query: 검색할 쿼리 텍스트
        project_id: 프로젝트 식별자 (기본값: "default")
        limit: 결과 개수 제한 (기본값: 5)
    
    Returns:
        검색 결과 목록
    """
    # vector_service.search_similar_prompts와 동일한 로직 사용
    await ensure_services_initialized()
    
    try:
        similar_prompts = await vector_service.search_similar_prompts(query, project_id, limit)
        
        return {
            "success": True,
            "query": query,
            "project_id": project_id,
            "results": similar_prompts,
            "total_results": len(similar_prompts)
        }
    except Exception as e:
        logger.error(f"유사 대화 검색 중 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "project_id": project_id,
            "results": []
        }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="대화 패턴 분석 중 오류가 발생했습니다."
)
@measure_performance(operation_name="analyze_conversation_patterns", threshold=5.0)
async def analyze_conversation_patterns(
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    대화 패턴을 분석하고 인사이트를 제공합니다.
    
    Args:
        project_id: 프로젝트 식별자 (기본값: "default")
    
    Returns:
        패턴 분석 결과
    """
    logger.info(f"대화 패턴 분석 요청: {project_id}")
    
    # 현재는 기본적인 분석만 제공
    # 향후 실제 패턴 분석 로직 구현 가능
    
    return {
        "success": True,
        "message": "패턴 분석 기능은 현재 개발 중입니다",
        "suggestion": "더 많은 대화 데이터가 필요합니다",
        "project_id": project_id,
        "analysis_date": "2024-12-19"
    }

# REMOVED: comprehensive_project_indexing 도구 제거됨 (자동 인덱싱으로 대체)


@mcp.tool()
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "고속 인덱싱 통계 조회 실패"}
)
@measure_performance(operation_name="get_fast_indexing_stats", threshold=3.0)
async def get_fast_indexing_stats() -> Dict[str, Any]:
    """
    고속 인덱싱 서비스의 성능 통계를 반환합니다.
    
    Returns:
        성능 설정 및 통계 정보
    """
    await ensure_services_initialized()
    
    stats = fast_indexing_service.get_performance_stats()
    
    return {
        "success": True,
        "performance_settings": stats,
        "optimization_features": [
            "병렬 파일 처리 (최대 20개 동시)",
            "배치 임베딩 생성 (10개씩)",
            "파일 해시 캐싱",
            "더 큰 청크 크기 (2KB)",
            "비동기 파일 I/O",
            "Thread Pool 사용",
            "스마트 파일 필터링"
        ],
        "speed_improvements": {
            "concurrent_processing": "5-10x faster",
            "batch_embeddings": "3-5x fewer API calls",
            "file_caching": "Skip already processed files",
            "larger_chunks": "Fewer storage operations"
        }
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프로젝트 파일 검색 중 오류가 발생했습니다."
)
@measure_performance(operation_name="search_project_files", threshold=3.0)
async def search_project_files(
    query: str,
    project_id: str = "default",
    file_type: str = "all",
    limit: int = 10
) -> Dict[str, Any]:
    """
    프로젝트 파일 내용에서 검색합니다.
    
    Args:
        query: 검색할 내용
        project_id: 프로젝트 식별자 (기본값: "default")
        file_type: 파일 타입 필터 ("code", "documentation", "all")
        limit: 결과 개수 제한 (기본값: 10)
    
    Returns:
        검색 결과
    """
    logger.info(f"프로젝트 파일 검색 요청: {query} (타입: {file_type})")
    
    await ensure_services_initialized()
    
    # 프로젝트 내 파일 검색
    results = await vector_service.search_similar_prompts(
        query=query,
        project_id=project_id,
        limit=limit
    )
    
    # 파일 컨텐츠만 필터링
    file_results = [
        result for result in results 
        if result.get('metadata', {}).get('is_file_content', False)
    ]
    
    # 파일 타입 필터링
    if file_type != "all":
        file_results = [
            result for result in file_results 
            if result.get('metadata', {}).get('file_type', '') == file_type
        ]
    
    formatted_results = []
    for result in file_results:
        metadata = result.get('metadata', {})
        formatted_results.append({
            "file_path": metadata.get('file_path', ''),
            "file_name": metadata.get('file_name', ''),
            "file_extension": metadata.get('file_extension', ''),
            "content_preview": result.get('content', '')[:200] + "...",
            "similarity": result.get('similarity', 0),
            "chunk_index": metadata.get('chunk_index', 0),
            "file_type": metadata.get('file_type', 'unknown')
        })
    
    return {
        "success": True,
        "query": query,
        "project_id": project_id,
        "file_type": file_type,
        "results_count": len(formatted_results),
        "results": formatted_results
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프로젝트 컨텍스트 정보 조회 중 오류가 발생했습니다."
)
@measure_performance(operation_name="get_project_context_info", threshold=3.0)
async def get_project_context_info(
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    프로젝트 컨텍스트 정보를 조회합니다.
    
    Args:
        project_id: 프로젝트 식별자 (기본값: "default")
    
    Returns:
        프로젝트 컨텍스트 정보
    """
    logger.info(f"프로젝트 컨텍스트 조회: {project_id}")
    
    await ensure_services_initialized()
    
    context = await vector_service.get_project_context(project_id)
    
    if context:
        return {
            "success": True,
            "project_id": project_id,
            "context": context
        }
    else:
        return {
            "success": False,
            "message": f"프로젝트 '{project_id}'의 컨텍스트를 찾을 수 없습니다.",
            "suggestion": "먼저 프로젝트 파일을 인덱싱해주세요."
        }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"status": "error", "message": "서버 상태 확인 실패"}
)
async def get_server_status() -> Dict[str, Any]:
    """
    서버 상태 정보를 반환합니다.
    
    Returns:
        서버 상태 정보
    """
    await ensure_services_initialized()
    return {
        "status": "running",
        "server_name": "Prompt Enhancement MCP Server",
        "version": "1.0.0",
        "services": {
            "vector_service": vector_service is not None,
            "enhancement_service": enhancement_service is not None,
            "file_indexing_service": file_indexing_service is not None,
            "fast_indexing_service": fast_indexing_service is not None,
            "analytics_service": analytics_service is not None
        },
        "transport": "SSE",
        "capabilities": [
            "prompt_enhancement",
            "vector_search",
            "conversation_storage",
            "pattern_analysis",
            "file_indexing",
            "fast_parallel_indexing",
            "project_analysis",
            "advanced_analytics",
            "clustering_analysis",
            "keyword_extraction",
            "trend_analysis"
        ]
    }

# 고급 분석 도구들

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프롬프트 패턴 분석 중 오류가 발생했습니다."
)
@measure_performance(operation_name="analyze_prompt_patterns", threshold=5.0)
async def analyze_prompt_patterns(
    project_id: str = "default",
    n_clusters: int = 5
) -> Dict[str, Any]:
    """
    프로젝트의 프롬프트 패턴을 클러스터링으로 분석합니다.
    
    Args:
        project_id: 프로젝트 식별자 (기본값: "default")
        n_clusters: 클러스터 개수 (기본값: 5)
    
    Returns:
        클러스터링 분석 결과
    """
    logger.info(f"프롬프트 패턴 분석 시작: {project_id}")
    
    await ensure_services_initialized()
    
    # 프로젝트의 모든 프롬프트 가져오기
    prompts = await vector_service.search_similar_prompts(
        query="",  # 빈 쿼리로 모든 프롬프트 검색
        project_id=project_id,
        limit=100
    )
    
    if len(prompts) < n_clusters:
        return {
            "success": False,
            "message": f"분석을 위해서는 최소 {n_clusters}개의 프롬프트가 필요합니다. 현재: {len(prompts)}개"
        }
    
    # 임베딩과 텍스트 추출
    embeddings = []
    texts = []
    for prompt in prompts:
        # 임베딩이 있으면 사용, 없으면 더미 임베딩
        embedding = prompt.get('embedding', [0.0] * 1536)  # OpenAI 기본 차원
        embeddings.append(embedding)
        texts.append(prompt.get('content', ''))
    
    # 클러스터링 수행
    clustering_result = await analytics_service.cluster_prompts(
        prompt_embeddings=embeddings,
        prompt_texts=texts,
        n_clusters=n_clusters
    )
    
    return {
        "success": True,
        "project_id": project_id,
        "total_prompts": len(prompts),
        "clustering_result": clustering_result
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="키워드 추출 중 오류가 발생했습니다."
)
@measure_performance(operation_name="extract_prompt_keywords", threshold=5.0)
async def extract_prompt_keywords(
    project_id: str = "default",
    max_features: int = 20
) -> Dict[str, Any]:
    """
    프로젝트 프롬프트에서 중요한 키워드를 TF-IDF로 추출합니다.
    
    Args:
        project_id: 프로젝트 식별자 (기본값: "default")
        max_features: 추출할 최대 키워드 수 (기본값: 20)
    
    Returns:
        키워드 추출 결과
    """
    logger.info(f"키워드 추출 시작: {project_id}")
    
    await ensure_services_initialized()
    
    # 프로젝트의 모든 프롬프트 가져오기
    prompts = await vector_service.search_similar_prompts(
        query="",
        project_id=project_id,
        limit=100
    )
    
    if not prompts:
        return {
            "success": False,
            "message": "분석할 프롬프트가 없습니다."
        }
    
    # 텍스트 추출
    texts = [prompt.get('content', '') for prompt in prompts]
    
    # TF-IDF 특성 추출
    features_result = await analytics_service.extract_text_features(texts)
    
    return {
        "success": True,
        "project_id": project_id,
        "total_prompts": len(prompts),
        "keywords": features_result.get('top_features', [])[:max_features],
        "vocabulary_size": features_result.get('vocabulary_size', 0)
    }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="트렌드 분석 중 오류가 발생했습니다."
)
@measure_performance(operation_name="analyze_prompt_trends", threshold=5.0)
async def analyze_prompt_trends(
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    프로젝트의 프롬프트 트렌드를 분석합니다.
    
    Args:
        project_id: 프로젝트 식별자 (기본값: "default")
    
    Returns:
        트렌드 분석 결과
    """
    logger.info(f"트렌드 분석 시작: {project_id}")
    
    await ensure_services_initialized()
    
    # 프로젝트의 모든 프롬프트 가져오기
    prompts = await vector_service.search_similar_prompts(
        query="",
        project_id=project_id,
        limit=200
    )
    
    if not prompts:
        return {
            "success": False,
            "message": "분석할 프롬프트가 없습니다."
        }
    
    # 메타데이터를 포함한 프롬프트 데이터 구성
    prompt_data = []
    for prompt in prompts:
        prompt_data.append({
            "content": prompt.get('content', ''),
            "created_at": prompt.get('metadata', {}).get('created_at'),
            "prompt_type": prompt.get('metadata', {}).get('prompt_type')
        })
    
    # 트렌드 분석
    trends = await analytics_service.analyze_prompt_trends(prompt_data)
    
    return {
        "success": True,
        "project_id": project_id,
        "analysis_period": "전체 기간",
        "trends": trends
    }

# 프롬프트 템플릿들

@mcp.prompt()
def create_enhanced_prompt(topic: str, context: str = "") -> str:
    """
    주제와 컨텍스트를 기반으로 개선된 프롬프트를 생성합니다.
    
    Args:
        topic: 프롬프트 주제
        context: 추가 컨텍스트 정보
    
    Returns:
        개선된 프롬프트 템플릿
    """
    logger.info(f"프롬프트 템플릿 생성: {topic}")
    
    base_prompt = f"""
다음 주제에 대한 고품질 프롬프트를 작성해주세요:

주제: {topic}
"""
    
    if context:
        base_prompt += f"\n컨텍스트: {context}\n"
    
    base_prompt += """
요구사항:
1. 명확하고 구체적인 지시사항
2. 예상 결과물 명시
3. 품질 기준 포함
4. 단계별 접근 방법 제시

위 요구사항을 만족하는 프롬프트를 생성해주세요.
"""
    
    return base_prompt

# 리소스들

@mcp.resource("prompt-history://projects/{project_id}")
@handle_errors(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.MEDIUM,
    user_message="프롬프트 히스토리 조회 중 오류가 발생했습니다."
)
@measure_performance(operation_name="get_prompt_history", threshold=3.0)
async def get_prompt_history(project_id: str) -> str:
    """
    특정 프로젝트의 프롬프트 히스토리를 반환합니다.
    
    Args:
        project_id: 프로젝트 식별자
    
    Returns:
        프롬프트 히스토리 정보
    """
    logger.info(f"프롬프트 히스토리 요청: {project_id}")
    
    await ensure_services_initialized()
    
    # 최근 프롬프트들 검색
    results = await vector_service.search_similar_prompts(
        query=project_id,  # 프로젝트 ID로 검색
        project_id=project_id,
        limit=10
    )
    
    history = f"프로젝트 '{project_id}'의 프롬프트 히스토리:\n\n"
    
    for i, result in enumerate(results, 1):
        history += f"{i}. {result.get('prompt', '')[:100]}...\n"
        history += f"   시간: {result.get('timestamp', 'N/A')}\n"
        history += f"   타입: {result.get('prompt_type', 'N/A')}\n\n"
    
    return history

@mcp.resource("server-info://status")
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "서버 정보 조회 실패"}
)
async def get_server_info() -> str:
    """
    서버 정보를 반환합니다.
    
    Returns:
        서버 정보 문자열
    """
    try:
        status = await get_server_status()
        
        info = f"""
MCP 서버 정보:
- 이름: {status.get('server_name', 'N/A')}
- 버전: {status.get('version', 'N/A')}
- 상태: {status.get('status', 'N/A')}
- 전송 방식: {status.get('transport', 'N/A')}

지원 기능:
"""
        
        for capability in status.get('capabilities', []):
            info += f"- {capability}\n"
        
        return info
        
    except Exception as e:
        logger.error(f"서버 정보 조회 실패: {e}")
        return f"서버 정보 조회 실패: {str(e)}"

# 🔄 피드백 관련 MCP 툴들
@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="피드백 제출 중 오류가 발생했습니다."
)
@measure_performance(operation_name="submit_user_feedback", threshold=3.0)
async def submit_user_feedback(
    enhancement_id: str,
    original_prompt: str,
    enhanced_prompt: str,
    project_id: str = "default",
    feedback_type: str = "accept",
    user_rating: int = None,
    user_comment: str = None,
    execution_success: bool = False,
    code_accepted: bool = False,
    time_to_success: float = None
) -> Dict[str, Any]:
    """
    사용자 피드백 제출
    
    Args:
        enhancement_id: 개선된 프롬프트 ID
        original_prompt: 원본 프롬프트
        enhanced_prompt: 개선된 프롬프트
        project_id: 프로젝트 ID
        feedback_type: 피드백 타입 (accept, reject, partial_accept, modify)
        user_rating: 사용자 평점 (1-5)
        user_comment: 사용자 코멘트
        execution_success: 실행 성공 여부
        code_accepted: 코드 수락 여부
        time_to_success: 성공까지 걸린 시간 (초)
    
    Returns:
        피드백 분석 결과
    """
    await ensure_services_initialized()
    
    try:
        from models.prompt_models import UserFeedback, FeedbackType
        
        # 피드백 객체 생성
        feedback = UserFeedback(
            enhancement_id=enhancement_id,
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            project_id=project_id,
            feedback_type=FeedbackType(feedback_type),
            user_rating=user_rating,
            user_comment=user_comment,
            execution_success=execution_success,
            code_accepted=code_accepted,
            time_to_success=time_to_success
        )
        
        # 피드백 처리
        analysis = await feedback_service.submit_feedback(feedback)
        
        return {
            "status": "success",
            "message": "피드백이 성공적으로 제출되었습니다",
            "analysis": {
                "enhancement_id": analysis.enhancement_id,
                "original_score": analysis.original_score,
                "adjusted_score": analysis.feedback_adjusted_score,
                "impact": analysis.feedback_impact,
                "recommendation": analysis.recommendation
            }
        }
        
    except Exception as e:
        logger.error(f"피드백 제출 오류: {e}")
        return {
            "status": "error",
            "message": f"피드백 제출 중 오류가 발생했습니다: {str(e)}"
        }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="피드백 통계 조회 중 오류가 발생했습니다."
)
@measure_performance(operation_name="get_feedback_statistics", threshold=3.0)
async def get_feedback_statistics(
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    프로젝트별 피드백 통계 조회
    
    Args:
        project_id: 프로젝트 ID
    
    Returns:
        피드백 통계 정보
    """
    await ensure_services_initialized()
    
    try:
        stats = await feedback_service.get_project_feedback_stats(project_id)
        return {
            "status": "success",
            "project_id": project_id,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"피드백 통계 조회 오류: {e}")
        return {
            "status": "error",
            "message": f"피드백 통계 조회 중 오류가 발생했습니다: {str(e)}"
        }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="피드백 패턴 분석 중 오류가 발생했습니다."
)
@measure_performance(operation_name="analyze_feedback_patterns", threshold=5.0)
async def analyze_feedback_patterns(
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    프로젝트별 피드백 패턴 분석
    
    Args:
        project_id: 프로젝트 ID
    
    Returns:
        피드백 패턴 분석 결과
    """
    await ensure_services_initialized()
    
    try:
        patterns = await feedback_service.analyze_project_patterns(project_id)
        return {
            "status": "success",
            "project_id": project_id,
            "patterns": patterns
        }
        
    except Exception as e:
        logger.error(f"피드백 패턴 분석 오류: {e}")
        return {
            "status": "error",
            "message": f"피드백 패턴 분석 중 오류가 발생했습니다: {str(e)}"
        }

@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="프롬프트 추천 조회 중 오류가 발생했습니다."
)
@measure_performance(operation_name="get_prompt_recommendations", threshold=3.0)
async def get_prompt_recommendations(
    prompt: str,
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    프롬프트에 대한 추천사항 조회
    
    Args:
        prompt: 분석할 프롬프트
        project_id: 프로젝트 ID
    
    Returns:
        프롬프트 추천사항
    """
    await ensure_services_initialized()
    
    try:
        recommendations = await feedback_service.get_recommendations_for_prompt(prompt, project_id)
        return {
            "status": "success",
            "prompt": prompt,
            "project_id": project_id,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"프롬프트 추천 조회 오류: {e}")
        return {
            "status": "error",
            "message": f"프롬프트 추천 조회 중 오류가 발생했습니다: {str(e)}"
        }


# 테스트 스캐폴딩 MCP 도구
@mcp.tool()
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="테스트 스캐폴딩 생성 중 오류가 발생했습니다."
)
@measure_performance(operation_name="generate_test_skeleton", threshold=3.0)
async def generate_test_skeleton(
    feature: str,
    framework: str = "pytest",
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    기능 설명과 테스트 프레임워크에 맞는 최소 실패 테스트 스켈레톤을 생성합니다.
    LLM 사용 가능 시 컨텍스트를 반영해 생성하고, 없으면 템플릿 기반으로 반환합니다.
    """
    await ensure_services_initialized()

    framework = (framework or "pytest").lower()

    # 프레임워크별 기본 템플릿
    templates = {
        "pytest": f"""
import pytest

def test_{'_' .join(feature.strip().split())}_should_fail_initially():
    # Arrange
    # TODO: setup

    # Act
    # TODO: call function under test

    # Assert
    with pytest.raises(AssertionError):
        assert False, "Write minimal failing assertion for: {feature}"
""".strip(),
        "jest": f"""
describe('{feature}', () => {{
  test('should fail initially', () => {{
    // Arrange

    // Act

    // Assert
    expect(false).toBe(true); // minimal failing expectation
  }});
}});
""".strip(),
        "unittest": f"""
import unittest

class Test{''.join([w.capitalize() for w in feature.split()])}(unittest.TestCase):
    def test_should_fail_initially(self):
        self.assertTrue(False, "Write minimal failing assertion for: {feature}")

if __name__ == '__main__':
    unittest.main()
""".strip(),
    }

    # LLM 사용 시 컨텍스트 반영 생성 시도
    try:
        if getattr(langchain_rag_service, "llm", None):
            context_info = await vector_service.get_project_context(project_id)
            prompt = (
                "Generate a minimal failing test skeleton for the following feature, "
                f"using {framework}. Include only the test file content, no explanations.\n\n"
                f"Feature: {feature}\n\nProject Context:\n{context_info or {}}\n"
            )
            generated = await langchain_rag_service.llm.arun(prompt=prompt)
            return {"success": True, "framework": framework, "content": generated}
    except Exception as e:
        logger.warning(f"LLM 기반 테스트 스캐폴딩 실패: {e}")

    # 폴백: 템플릿 반환
    content = templates.get(framework, templates["pytest"]) 
    return {"success": True, "framework": framework, "content": content}


# 시스템 검증 엔드포인트
@mcp.custom_route(path="/api/v1/validate", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "검증 실패", "success": False}
)
@measure_performance(operation_name="validate_system", threshold=3.0)
async def validate_system(request):
    """서버/인덱싱/LLM/에러/성능 상태를 통합 검증합니다."""
    await ensure_services_initialized()

    if not _check_api_key(request) or not _check_jwt(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _check_rate_limit(request):
        return JSONResponse({"error": "Too Many Requests", "success": False}, status_code=429)

    # 쿼리 파라미터
    params = request.query_params
    project_id = params.get("project_id", "default")

    if REQUEST_COUNT:
        REQUEST_COUNT.labels('/api/v1/validate').inc()
    with _RequestTimer('/api/v1/validate'):
        server = await get_server_status()
        indexing = await vector_service.get_search_statistics(project_id)
        errors = error_handler.get_error_stats()
        performance = error_handler.get_performance_stats()
    llm_available = getattr(langchain_rag_service, "llm", None) is not None

    return JSONResponse({
        "success": True,
        "llm_available": llm_available,
        "server": server,
        "indexing": indexing,
        "errors": errors,
        "performance": performance
    })

@mcp.custom_route(path="/metrics", methods=["GET"])
async def metrics_endpoint(request):
    """Prometheus metrics (no audit/logging/instrumentation)"""
    if not _PROM_AVAILABLE:
        return Response(content="Prometheus not available", media_type="text/plain", status_code=501)
    if REQUEST_COUNT:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# 인덱스 워밍업 API
@mcp.custom_route(path="/api/v1/index/warmup/{project_id}", methods=["POST"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "워밍업 실패", "success": False}
)
@measure_performance(operation_name="warmup_indices", threshold=10.0)
async def warmup_indices(request):
    await ensure_services_initialized()
    if not _check_rate_limit(request):
        return JSONResponse({"error": "Too Many Requests"}, status_code=429)
    project_id = request.path_params.get("project_id", "default")
    result = await vector_service.warmup_project_indices(project_id)
    return JSONResponse(result)


# 간단 HTML 대시보드
@mcp.custom_route(path="/dashboard", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error=Response(content="<h1>Error</h1>")
)
@measure_performance(operation_name="dashboard", threshold=1.0)
async def dashboard(request):
    await ensure_services_initialized()
    server = await get_server_status()
    errors = error_handler.get_error_stats()
    perf = error_handler.get_performance_stats()
    # 최근 감사 로그 10개 로드
    recent = await _read_recent_audit_items(limit=10)
    rows = "".join(
        f"<tr><td>{i+1}</td><td><code>{item.get('event','')}</code></td><td>{item.get('timestamp','')}</td><td><code>{(item.get('meta') or {}).get('project_id','')}</code></td></tr>"
        for i, item in enumerate(recent)
    )
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>MCP Server Dashboard</title>
  <style>
    body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .col {{ flex: 1 1 280px; }}
    h1 {{ margin-top: 0; }}
    code {{ background:#f3f4f6; padding:2px 4px; border-radius:4px; }}
  </style>
  <meta http-equiv="refresh" content="5"/>
  </head>
<body>
  <h1>MCP Server Dashboard</h1>
  <div class="row">
    <div class="card col">
      <h3>Server</h3>
      <div>Name: <code>{server.get('server_name')}</code></div>
      <div>Status: <b>{server.get('status')}</b></div>
      <div>Version: {server.get('version')}</div>
    </div>
    <div class="card col">
      <h3>Performance</h3>
      <div>Avg Response Time: {perf.get('avg_response_time'):.3f}s</div>
      <div>Total Ops: {perf.get('total_operations')}</div>
      <div>Slow Ops: {len(perf.get('slow_operations', []))}</div>
    </div>
    <div class="card col">
      <h3>Errors</h3>
      <div>Total: {errors.get('total_errors')}</div>
      <div>Recent: {len(errors.get('recent_errors', []))}</div>
    </div>
  </div>
  <div class="card">
    <h3>Links</h3>
    <ul>
      <li><a href="/api/v1/validate">/api/v1/validate</a></li>
      <li><a href="/metrics">/metrics</a> (if available)</li>
      <li><a href="/api/v1/audit/recent">/api/v1/audit/recent</a></li>
    </ul>
  </div>
  <div class="card">
    <h3>Recent Audit</h3>
    <table border="1" cellpadding="6" cellspacing="0">
      <thead>
        <tr><th>#</th><th>Event</th><th>Timestamp</th><th>Project</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</body>
</html>
"""
    return Response(content=html, media_type="text/html")


# 감사 로그 조회 API (최근 N개)
@mcp.custom_route(path="/api/v1/audit/recent", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "감사 로그 조회 실패", "success": False}
)
@measure_performance(operation_name="get_recent_audit", threshold=2.0)
async def get_recent_audit(request):
    limit = int(request.query_params.get("limit", 100))
    project_id = request.query_params.get("project_id")
    # 프로젝트 지정 시 해당 파일 우선
    target_path = _project_audit_path(project_id) if project_id else audit_log_path
    # 파일이 없으면 비어있는 결과 반환
    if not os.path.exists(target_path):
        return JSONResponse({"success": True, "items": []})
    items = []
    try:
        # 최근 N라인만 읽는 간단한 방법 (큰 파일 고려 시 tail 방식 대체 가능)
        async with aiofiles.open(target_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        items = items[-limit:]
        return JSONResponse({"success": True, "items": items})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@mcp.custom_route(path="/api/v1/audit/search", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "감사 로그 검색 실패", "success": False}
)
@measure_performance(operation_name="search_audit", threshold=3.0)
async def search_audit(request):
    """감사 로그 검색(프로젝트/이벤트/시작시간 기준 필터)"""
    project_id = request.query_params.get("project_id")
    event_filter = request.query_params.get("event")
    since = request.query_params.get("since")  # ISO8601
    limit = int(request.query_params.get("limit", 200))

    target_path = _project_audit_path(project_id) if project_id else audit_log_path
    if not os.path.exists(target_path):
        return JSONResponse({"success": True, "items": []})

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except Exception:
            since_dt = None

    items = []
    try:
        async with aiofiles.open(target_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if event_filter and event_filter not in str(obj.get('event', '')):
                    continue
                if since_dt:
                    try:
                        ts = datetime.fromisoformat(str(obj.get('timestamp', '')).replace("Z", "+00:00"))
                        if ts < since_dt:
                            continue
                    except Exception:
                        pass
                items.append(obj)
        # 최신순으로 제한
        items = items[-limit:]
        return JSONResponse({"success": True, "items": items})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def _read_recent_audit_items(project_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    target_path = _project_audit_path(project_id) if project_id else audit_log_path
    if not os.path.exists(target_path):
        return []
    items: List[Dict[str, Any]] = []
    try:
        async with aiofiles.open(target_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items[-limit:]
    except Exception:
        return []


# 🔄 피드백 관련 FastAPI 엔드포인트들
@mcp.custom_route(path="/api/v1/feedback", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="피드백 제출 중 오류가 발생했습니다."
)
@measure_performance(operation_name="submit_feedback", threshold=3.0)
async def submit_feedback_endpoint(request):
    """사용자 피드백 제출"""
    await ensure_services_initialized()
    
    try:
        # 요청 데이터 파싱
        data = await request.json()
        
        # UserFeedback 객체 생성
        from models.prompt_models import UserFeedback, FeedbackType
        feedback = UserFeedback(
            enhancement_id=data.get("enhancement_id"),
            original_prompt=data.get("original_prompt"),
            enhanced_prompt=data.get("enhanced_prompt"),
            project_id=data.get("project_id", "default"),
            feedback_type=FeedbackType(data.get("feedback_type")),
            user_rating=data.get("user_rating"),
            user_comment=data.get("user_comment"),
            execution_success=data.get("execution_success", False),
            code_accepted=data.get("code_accepted", False),
            time_to_success=data.get("time_to_success")
        )
        
        # 피드백 처리
        analysis = await feedback_service.submit_feedback(feedback)
        
        return {
            "status": "success",
            "message": "피드백이 성공적으로 제출되었습니다",
            "analysis": {
                "enhancement_id": analysis.enhancement_id,
                "original_score": analysis.original_score,
                "adjusted_score": analysis.feedback_adjusted_score,
                "impact": analysis.feedback_impact,
                "recommendation": analysis.recommendation
            }
        }
        
    except Exception as e:
        logger.error(f"피드백 제출 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.custom_route(path="/api/v1/feedback/stats/{project_id}", methods=["GET"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="피드백 통계 조회 중 오류가 발생했습니다."
)
@measure_performance(operation_name="get_feedback_stats", threshold=3.0)
async def get_feedback_stats_endpoint(request):
    """프로젝트별 피드백 통계"""
    await ensure_services_initialized()
    
    # Path parameter 추출
    project_id = request.path_params.get("project_id")
    
    try:
        stats = await feedback_service.get_project_feedback_stats(project_id)
        return {
            "status": "success",
            "project_id": project_id,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"피드백 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🎯 LangChain RAG 기반 엔드포인트들

@mcp.custom_route(path="/api/v1/rag/enhance-prompt", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="RAG 기반 프롬프트 개선 중 오류가 발생했습니다."
)
@measure_performance(operation_name="rag_enhance_prompt", threshold=10.0)
async def rag_enhance_prompt(request):
    """RAG 기반 프롬프트 개선 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request) or not _check_jwt(request):
            return JSONResponse({"error": "Unauthorized", "success": False}, status_code=401)
        if not _check_rate_limit(request):
            return JSONResponse({"error": "Too Many Requests", "success": False}, status_code=429)
        
        # 요청 데이터 파싱
        data = await request.json()
        user_prompt = data.get("prompt", "")
        project_id = data.get("project_id", "default")
        context_limit = data.get("context_limit", 5)
        
        # 입력 검증
        if not user_prompt.strip():
            return JSONResponse({"error": "프롬프트가 비어있습니다", "success": False})
        
        # 프로젝트 쿼터 체크 및 감사
        if not _check_project_quota(project_id):
            return JSONResponse({"error": "Project quota exceeded", "success": False}, status_code=429)
        _audit('rag_enhance_prompt', {"project_id": project_id})

        # RAG 기반 프롬프트 향상
        if REQUEST_COUNT:
            REQUEST_COUNT.labels('/api/v1/rag/enhance-prompt').inc()
        with _RequestTimer('/api/v1/rag/enhance-prompt'):
            result = await langchain_rag_service.generate_enhanced_prompt(
                user_prompt=user_prompt,
                project_id=project_id,
                context_limit=context_limit
            )
            return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"RAG 기반 프롬프트 개선 중 오류: {str(e)}")
        return JSONResponse({"error": str(e), "success": False})

@mcp.custom_route(path="/api/v1/rag/generate-code", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="RAG 기반 코드 생성 중 오류가 발생했습니다."
)
@measure_performance(operation_name="rag_generate_code", threshold=15.0)
async def rag_generate_code(request):
    """RAG 기반 코드 생성 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request) or not _check_jwt(request):
            return JSONResponse({"error": "Unauthorized", "success": False}, status_code=401)
        if not _check_rate_limit(request):
            return {"error": "Too Many Requests", "success": False, "status_code": 429}
        
        # 요청 데이터 파싱
        data = await request.json()
        user_prompt = data.get("prompt", "")
        project_id = data.get("project_id", "default")
        context_limit = data.get("context_limit", 5)
        
        # 입력 검증
        if not user_prompt.strip():
            return {"error": "프롬프트가 비어있습니다", "success": False}
        
        # 스트리밍 여부
        stream = bool(data.get("stream", False))

        # 프로젝트 쿼터 체크 및 감사
        if not _check_project_quota(project_id):
            return JSONResponse({"error": "Project quota exceeded", "success": False}, status_code=429)
        _audit('rag_generate_code', {"project_id": project_id, "stream": stream})

        # RAG 기반 코드 생성
        if not stream:
            if REQUEST_COUNT:
                REQUEST_COUNT.labels('/api/v1/rag/generate-code').inc()
            with _RequestTimer('/api/v1/rag/generate-code'):
                result = await langchain_rag_service.generate_code_with_rag(
                    user_prompt=user_prompt,
                    project_id=project_id,
                    context_limit=context_limit
                )
                return JSONResponse(result)
        
        async def event_generator():
            # 컨텍스트 준비
            enhanced = await langchain_rag_service.generate_enhanced_prompt(
                user_prompt=user_prompt,
                project_id=project_id,
                context_limit=context_limit
            )
            yield await create_sse_event(SSEEventType.ENHANCEMENT_COMPLETE, {"enhanced": True})
            # LLM 스트리밍 (가능 시)
            if getattr(langchain_rag_service, 'llm', None) and hasattr(langchain_rag_service.llm, 'astream'):
                async for chunk in langchain_rag_service.llm.astream({
                    'context': enhanced.get('context', ''),
                    'question': user_prompt
                }):
                    yield await create_sse_event(SSEEventType.ENHANCEMENT_PROGRESS, {"delta": chunk})
            else:
                # 폴백: 전체 응답 한 번에
                result = await langchain_rag_service.generate_code_with_rag(
                    user_prompt=user_prompt,
                    project_id=project_id,
                    context_limit=context_limit
                )
                yield await create_sse_event(SSEEventType.ENHANCEMENT_COMPLETE, result)

        return EventSourceResponse(event_generator())
        
    except Exception as e:
        logger.error(f"RAG 기반 코드 생성 중 오류: {str(e)}")
        return JSONResponse({"error": str(e), "success": False}, status_code=500)

@mcp.custom_route(path="/api/v1/rag/search-summarize", methods=["POST"])
@handle_errors(
    category=ErrorCategory.AI_SERVICE,
    level=ErrorLevel.MEDIUM,
    user_message="RAG 기반 검색 및 요약 중 오류가 발생했습니다."
)
@measure_performance(operation_name="rag_search_summarize", threshold=10.0)
async def rag_search_summarize(request):
    """RAG 기반 검색 및 요약 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request):
            return JSONResponse({"error": "Unauthorized", "success": False}, status_code=401)
        if not _check_rate_limit(request):
            return JSONResponse({"error": "Too Many Requests", "success": False}, status_code=429)
        
        # 요청 데이터 파싱
        data = await request.json()
        query = data.get("query", "")
        project_id = data.get("project_id", "default")
        limit = data.get("limit", 3)
        
        # 입력 검증
        if not query.strip():
            return JSONResponse({"error": "검색 쿼리가 비어있습니다", "success": False}, status_code=400)
        
        # 프로젝트 쿼터 체크 및 감사
        if not _check_project_quota(project_id):
            return JSONResponse({"error": "Project quota exceeded", "success": False}, status_code=429)
        _audit('rag_search_summarize', {"project_id": project_id})

        # RAG 기반 검색 및 요약
        if REQUEST_COUNT:
            REQUEST_COUNT.labels('/api/v1/rag/search-summarize').inc()
        with _RequestTimer('/api/v1/rag/search-summarize'):
            result = await langchain_rag_service.search_and_summarize(
                query=query,
                project_id=project_id,
                limit=limit
            )
            return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"RAG 기반 검색 및 요약 중 오류: {str(e)}")
        return JSONResponse({"error": str(e), "success": False}, status_code=500)

@mcp.custom_route(path="/api/v1/resource/snippet", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "스니펫 조회 실패", "success": False}
)
@measure_performance(operation_name="get_snippet", threshold=1.0)
async def get_snippet(request):
    """코드/문서 스니펫 반환 (파일 경로 + 라인 범위)"""
    await ensure_services_initialized()
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _check_rate_limit(request):
        return JSONResponse({"error": "Too Many Requests"}, status_code=429)

    params = request.query_params
    file_path = params.get("file_path")
    start_line = int(params.get("start_line", 1))
    end_line = int(params.get("end_line", start_line + 200))
    if not file_path:
        return JSONResponse({"error": "file_path is required"}, status_code=400)

    snippet = await _read_file_snippet(file_path, start_line, end_line)
    language = _guess_language(file_path)
    return JSONResponse({"success": True, "language": language, **snippet})

# 🎯 파일 와처 기반 엔드포인트들

@mcp.custom_route(path="/api/v1/watcher/start", methods=["POST"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.MEDIUM,
    user_message="파일 감시 시작 중 오류가 발생했습니다."
)
@measure_performance(operation_name="start_file_watcher", threshold=5.0)
async def start_file_watcher(request):
    """파일 감시 시작 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        
        # 요청 데이터 파싱
        data = await request.json()
        project_path = data.get("project_path", "")
        project_id = data.get("project_id", "default")
        recursive = data.get("recursive", True)
        auto_upload = data.get("auto_upload", True)
        
        # 입력 검증
        if not project_path.strip():
            return {"error": "프로젝트 경로가 비어있습니다", "success": False}
        
        # 파일 감시 시작
        result = await file_watcher_service.start_watching_project(
            project_path=project_path,
            project_id=project_id,
            recursive=recursive,
            auto_upload=auto_upload
        )
        
        return result
        
    except Exception as e:
        logger.error(f"파일 감시 시작 중 오류: {str(e)}")
        return {"error": str(e), "success": False}

@mcp.custom_route(path="/api/v1/watcher/stop", methods=["POST"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.MEDIUM,
    user_message="파일 감시 중지 중 오류가 발생했습니다."
)
@measure_performance(operation_name="stop_file_watcher", threshold=3.0)
async def stop_file_watcher(request):
    """파일 감시 중지 엔드포인트"""
    try:
        await ensure_services_initialized()
        if not _check_api_key(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        
        # 요청 데이터 파싱
        data = await request.json()
        project_id = data.get("project_id", "default")
        
        # 입력 검증
        if not project_id.strip():
            return {"error": "프로젝트 ID가 비어있습니다", "success": False}
        
        # 파일 감시 중지
        result = await file_watcher_service.stop_watching_project(project_id)
        
        return result
        
    except Exception as e:
        logger.error(f"파일 감시 중지 중 오류: {str(e)}")
        return {"error": str(e), "success": False}

@mcp.custom_route(path="/api/v1/watcher/status", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"error": "감시 상태 조회 실패", "success": False}
)
@measure_performance(operation_name="get_watcher_status", threshold=3.0)
async def get_watcher_status(request):
    """파일 감시 상태 조회 엔드포인트"""
    try:
        await ensure_services_initialized()
        
        # 감시 상태 조회
        result = await file_watcher_service.get_watching_status()
        
        return result
        
    except Exception as e:
        logger.error(f"감시 상태 조회 중 오류: {str(e)}")
        return {"error": str(e), "success": False}

# 헬스체크 엔드포인트
@mcp.custom_route(path="/api/v1/heartbeat", methods=["GET"])
@handle_errors(
    category=ErrorCategory.SYSTEM,
    level=ErrorLevel.LOW,
    return_on_error={"status": "unhealthy", "message": "헬스체크 실패"}
)
async def heartbeat(request):
    """Docker 헬스체크용 엔드포인트"""
    try:
        # 서비스 상태 확인
        status = await get_server_status()
        return JSONResponse({
            "status": "healthy",
            "message": "MCP 서버가 정상 작동 중입니다",
            "services": status.get("services", {}),
            "timestamp": asyncio.get_event_loop().time()
        })
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        return JSONResponse({
            "status": "unhealthy", 
            "message": f"서버 오류: {str(e)}",
            "timestamp": asyncio.get_event_loop().time()
        })

# 주석 처리된 네트워크 업로드 함수 제거됨 (사용하지 않음)









# 서버 종료 시 자동 인덱싱 서비스 정리
async def cleanup_services():
    """서비스 정리"""
    logger.info("서비스 정리 시작...")
    
    try:
        # 자동 인덱싱 서비스 중지
        if auto_indexing_service and auto_indexing_service.is_running:
            logger.info("자동 인덱싱 서비스 중지 중...")
            await auto_indexing_service.stop()
        
        # 파일 감시 서비스 중지
        if file_watcher_service:
            logger.info("파일 감시 서비스 중지 중...")
            await file_watcher_service.stop_all_watchers()
        
        # 벡터 서비스 정리
        if vector_service:
            logger.info("벡터 서비스 정리 중...")
            if hasattr(vector_service, 'embeddings') and vector_service.embeddings:
                if hasattr(vector_service.embeddings, 'close'):
                    await vector_service.embeddings.close()
        
        # 고속 인덱싱 서비스 정리
        if fast_indexing_service:
            logger.info("고속 인덱싱 서비스 정리 중...")
            if hasattr(fast_indexing_service, 'thread_executor'):
                fast_indexing_service.thread_executor.shutdown(wait=False)
        
        # ChromaDB 연결 정리
        if vector_service and vector_service.chroma_client:
            logger.info("ChromaDB 연결 정리 중...")
            # ChromaDB는 자동으로 정리됨
        
        logger.info("모든 서비스가 정리되었습니다")
        
    except Exception as e:
        logger.error(f"서비스 정리 중 오류: {e}", exc_info=True)

if __name__ == "__main__":
    import signal
    import sys
    
    logger.info("🚀 FastMCP 서버 시작...")
    
    # 서비스 초기화
    try:
        asyncio.run(initialize_services())
        logger.info("서비스 초기화 완료")
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        sys.exit(1)
    
    def signal_handler(sig, frame):
        """신호 처리기"""
        logger.info(f"종료 신호 수신: {sig}")
        try:
            # 새로운 이벤트 루프 생성하여 정리
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cleanup_services())
            loop.close()
        except Exception as e:
            logger.error(f"서비스 정리 중 오류: {e}")
        sys.exit(0)
    
    # 신호 처리기 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # FastMCP 서버 실행 (SSE 모드)
        logger.info("SSE 모드로 MCP 서버 실행 중...")
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트 감지")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"서버 실행 중 오류: {e}", exc_info=True)
        sys.exit(1) 