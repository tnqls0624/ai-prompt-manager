from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class FeedbackType(str, Enum):
    """피드백 타입"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXCELLENT = "excellent"

class PerformanceMetric(BaseModel):
    """성능 지표"""
    response_time: float = Field(description="응답 시간 (초)")
    token_count: int = Field(description="토큰 수")
    confidence_score: float = Field(ge=0.0, le=1.0, description="신뢰도 점수")
    relevance_score: float = Field(ge=0.0, le=1.0, description="관련성 점수")
    clarity_score: float = Field(ge=0.0, le=1.0, description="명확성 점수")
    completeness_score: float = Field(ge=0.0, le=1.0, description="완성도 점수")
    context_utilization: float = Field(ge=0.0, le=1.0, description="컨텍스트 활용도")

class UserFeedback(BaseModel):
    """사용자 피드백"""
    id: str = Field(description="피드백 ID")
    prompt_id: str = Field(description="관련 프롬프트 ID")
    enhanced_prompt_id: str = Field(description="개선된 프롬프트 ID")
    user_id: str = Field(default="anonymous", description="사용자 ID")
    feedback_type: FeedbackType = Field(description="피드백 타입")
    rating: Optional[int] = Field(ge=1, le=5, description="평점 (1-5)")
    comment: Optional[str] = Field(description="피드백 코멘트")
    usage_count: int = Field(default=0, description="사용 횟수")
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0, description="성공률")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnhancedPromptResult(BaseModel):
    """고도화된 프롬프트 결과"""
    id: str = Field(description="결과 ID")
    original_prompt: str = Field(description="원본 프롬프트")
    enhanced_prompt: str = Field(description="개선된 프롬프트")
    project_id: str = Field(description="프로젝트 ID")
    
    # RAG 컨텍스트
    retrieved_contexts: List[Dict[str, Any]] = Field(default_factory=list, description="검색된 컨텍스트")
    context_relevance_scores: List[float] = Field(default_factory=list, description="컨텍스트 관련성 점수")
    
    # 성능 지표
    performance_metrics: PerformanceMetric = Field(description="성능 지표")
    
    # 개선 과정
    improvement_strategy: str = Field(description="개선 전략")
    reasoning: str = Field(description="개선 근거")
    confidence_factors: List[str] = Field(default_factory=list, description="신뢰도 요인")
    
    # 피드백 및 학습
    feedback_history: List[UserFeedback] = Field(default_factory=list, description="피드백 히스토리")
    learning_score: float = Field(ge=0.0, le=1.0, default=0.5, description="학습 점수")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ContextualKnowledge(BaseModel):
    """컨텍스트 지식"""
    id: str = Field(description="지식 ID")
    content: str = Field(description="지식 내용")
    knowledge_type: str = Field(description="지식 타입: pattern, solution, best_practice")
    project_patterns: List[str] = Field(default_factory=list, description="프로젝트 패턴")
    tech_stack_affinity: Dict[str, float] = Field(default_factory=dict, description="기술 스택 친화도")
    success_rate: float = Field(ge=0.0, le=1.0, description="성공률")
    usage_frequency: int = Field(default=0, description="사용 빈도")
    effectiveness_score: float = Field(ge=0.0, le=1.0, description="효과성 점수")
    created_at: datetime = Field(default_factory=datetime.now)

class SmartRecommendation(BaseModel):
    """스마트 추천"""
    id: str = Field(description="추천 ID")
    prompt_pattern: str = Field(description="프롬프트 패턴")
    recommended_improvement: str = Field(description="추천 개선사항")
    confidence: float = Field(ge=0.0, le=1.0, description="추천 신뢰도")
    evidence: List[str] = Field(default_factory=list, description="근거")
    applicable_contexts: List[str] = Field(default_factory=list, description="적용 가능한 컨텍스트")
    success_examples: List[str] = Field(default_factory=list, description="성공 사례")

class LearningInsight(BaseModel):
    """학습 인사이트"""
    id: str = Field(description="인사이트 ID")
    project_id: str = Field(description="프로젝트 ID")
    insight_type: str = Field(description="인사이트 타입")
    title: str = Field(description="인사이트 제목")
    description: str = Field(description="인사이트 설명")
    impact_score: float = Field(ge=0.0, le=1.0, description="영향도 점수")
    actionable_suggestions: List[str] = Field(default_factory=list, description="실행 가능한 제안")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="지원 데이터")
    created_at: datetime = Field(default_factory=datetime.now)

class IntelligentSearchQuery(BaseModel):
    """지능형 검색 쿼리"""
    original_query: str = Field(description="원본 쿼리")
    expanded_queries: List[str] = Field(default_factory=list, description="확장된 쿼리들")
    semantic_embeddings: List[List[float]] = Field(default_factory=list, description="의미적 임베딩")
    search_strategy: str = Field(description="검색 전략")
    filters: Dict[str, Any] = Field(default_factory=dict, description="검색 필터")
    ranking_weights: Dict[str, float] = Field(default_factory=dict, description="랭킹 가중치")

class AdvancedRAGRequest(BaseModel):
    """고도화된 RAG 요청"""
    query: str = Field(description="검색 쿼리")
    project_id: str = Field(description="프로젝트 ID")
    context_limit: int = Field(default=10, description="컨텍스트 제한")
    include_performance_data: bool = Field(default=True, description="성능 데이터 포함")
    personalization_level: float = Field(ge=0.0, le=1.0, default=0.7, description="개인화 수준")
    creativity_level: float = Field(ge=0.0, le=1.0, default=0.5, description="창의성 수준")
    safety_level: float = Field(ge=0.0, le=1.0, default=0.8, description="안전성 수준")

class AdvancedRAGResponse(BaseModel):
    """고도화된 RAG 응답"""
    enhanced_prompt: str = Field(description="개선된 프롬프트")
    confidence_score: float = Field(ge=0.0, le=1.0, description="신뢰도 점수")
    retrieved_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list, description="추론 체인")
    alternative_approaches: List[str] = Field(default_factory=list, description="대안 접근법")
    risk_factors: List[str] = Field(default_factory=list, description="위험 요소")
    optimization_suggestions: List[str] = Field(default_factory=list, description="최적화 제안")
    performance_prediction: PerformanceMetric = Field(description="성능 예측")
    learning_opportunities: List[LearningInsight] = Field(default_factory=list, description="학습 기회") 