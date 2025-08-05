from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class PromptType(str, Enum):
    """프롬프트 타입"""
    USER_QUERY = "user_query"
    AI_RESPONSE = "ai_response"
    SYSTEM_PROMPT = "system_prompt"
    ENHANCED_PROMPT = "enhanced_prompt"

class FeedbackType(str, Enum):
    """피드백 타입"""
    ACCEPT = "accept"
    REJECT = "reject"
    PARTIAL_ACCEPT = "partial_accept"
    MODIFY = "modify"

class ProjectContext(BaseModel):
    """프로젝트 컨텍스트"""
    project_id: str
    project_name: str
    description: Optional[str] = None
    tech_stack: List[str] = []
    file_patterns: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

class PromptHistory(BaseModel):
    """프롬프트 히스토리"""
    id: str
    project_id: str
    content: str
    prompt_type: PromptType
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    # 피드백 관련 필드 추가
    feedback_score: float = 0.0  # -1.0 ~ 1.0
    usage_count: int = 0
    success_rate: float = 0.0

class PromptRequest(BaseModel):
    """프롬프트 요청"""
    prompt: str
    context: Optional[str] = None
    project_type: Optional[str] = None

class PromptResponse(BaseModel):
    """프롬프트 응답"""
    original_prompt: str
    enhanced_prompt: str
    project_type: Optional[str] = None

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    version: str
    
class PromptEnhanceRequest(BaseModel):
    """프롬프트 개선 요청"""
    original_prompt: str
    project_id: str
    context_limit: int = Field(default=5, description="참조할 컨텍스트 수")
    include_docs: bool = Field(default=True, description="관련 문서 포함 여부")

class PromptEnhanceResponse(BaseModel):
    """프롬프트 개선 응답"""
    enhanced_prompt: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    context_used: List[str] = []
    suggestions: List[str] = []
    # 추적을 위한 ID 추가
    enhancement_id: str = Field(default_factory=lambda: f"enh_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

class UserFeedback(BaseModel):
    """사용자 피드백"""
    enhancement_id: str  # 개선된 프롬프트의 ID
    original_prompt: str
    enhanced_prompt: str
    project_id: str
    feedback_type: FeedbackType
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 별점")
    user_comment: Optional[str] = None
    execution_success: bool = Field(default=False, description="제안된 프롬프트로 성공적으로 코드 생성했는지")
    code_accepted: bool = Field(default=False, description="생성된 코드를 사용자가 수락했는지")
    time_to_success: Optional[float] = Field(None, description="성공까지 걸린 시간(초)")
    created_at: datetime = Field(default_factory=datetime.now)

class FeedbackAnalysis(BaseModel):
    """피드백 분석 결과"""
    enhancement_id: str
    original_score: float
    feedback_adjusted_score: float
    feedback_impact: float  # 피드백이 점수에 미친 영향
    recommendation: str  # 개선 권장사항

class MCPMessage(BaseModel):
    """MCP 메시지 포맷"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None 