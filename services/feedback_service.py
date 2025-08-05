"""
피드백 서비스 - 사용자 피드백을 처리하고 프롬프트 개선에 반영
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.prompt_models import (
    UserFeedback, FeedbackType, FeedbackAnalysis, PromptHistory
)
from services.vector_service import VectorService
from services.error_handler import error_handler, performance_monitor, handle_errors

logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """피드백 처리 및 학습 클래스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.feedback_history: Dict[str, List[UserFeedback]] = defaultdict(list)
        self.learning_weights = {
            FeedbackType.ACCEPT: 1.0,
            FeedbackType.PARTIAL_ACCEPT: 0.5,
            FeedbackType.REJECT: -1.0,
            FeedbackType.MODIFY: -0.3
        }
        
    @handle_errors()
    @performance_monitor()
    async def process_feedback(self, feedback: UserFeedback) -> FeedbackAnalysis:
        """피드백을 처리하고 점수 조정"""
        logger.info(f"Processing feedback for enhancement_id: {feedback.enhancement_id}")
        
        # 피드백 저장
        self.feedback_history[feedback.project_id].append(feedback)
        
        # 기존 점수 조회
        original_score = await self._get_original_score(feedback.enhancement_id)
        
        # 피드백 기반 점수 조정
        feedback_impact = self._calculate_feedback_impact(feedback)
        adjusted_score = self._adjust_score(original_score, feedback_impact)
        
        # 벡터 DB에 피드백 정보 업데이트
        await self._update_vector_db(feedback, adjusted_score)
        
        # 분석 결과 생성
        analysis = FeedbackAnalysis(
            enhancement_id=feedback.enhancement_id,
            original_score=original_score,
            feedback_adjusted_score=adjusted_score,
            feedback_impact=feedback_impact,
            recommendation=self._generate_recommendation(feedback, feedback_impact)
        )
        
        logger.info(f"Feedback processed: {feedback.feedback_type}, impact: {feedback_impact}")
        return analysis
    
    def _calculate_feedback_impact(self, feedback: UserFeedback) -> float:
        """피드백의 영향도 계산"""
        base_weight = self.learning_weights.get(feedback.feedback_type, 0.0)
        
        # 추가 가중치 계산
        multipliers = []
        
        # 별점 기반 가중치
        if feedback.user_rating:
            rating_weight = (feedback.user_rating - 3) / 2  # -1 ~ 1 범위로 정규화
            multipliers.append(rating_weight)
        
        # 코드 수락 여부
        if feedback.code_accepted:
            multipliers.append(0.5)
        elif feedback.execution_success:
            multipliers.append(0.2)
        else:
            multipliers.append(-0.3)
            
        # 성공 시간 기반 가중치
        if feedback.time_to_success:
            # 빠른 성공일수록 높은 점수
            time_weight = max(0.1, 1.0 - (feedback.time_to_success / 300))  # 5분 기준
            multipliers.append(time_weight)
        
        # 최종 영향도 계산
        final_multiplier = np.mean(multipliers) if multipliers else 1.0
        return base_weight * final_multiplier
    
    def _adjust_score(self, original_score: float, feedback_impact: float) -> float:
        """점수 조정"""
        # 학습률 적용
        from config import settings
        learning_rate = settings.feedback_learning_rate
        adjusted_score = original_score + (feedback_impact * learning_rate)
        
        # 점수 범위 제한 (-1.0 ~ 1.0)
        return max(-1.0, min(1.0, adjusted_score))
    
    async def _get_original_score(self, enhancement_id: str) -> float:
        """기존 점수 조회"""
        try:
            # 벡터 DB에서 해당 프롬프트의 점수 조회
            results = await self.vector_service.search_similar_content(
                query=enhancement_id,
                project_id="system",
                limit=1
            )
            
            if results and len(results) > 0:
                return results[0].get('feedback_score', 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting original score: {e}")
            return 0.0
    
    async def _update_vector_db(self, feedback: UserFeedback, adjusted_score: float):
        """벡터 DB 업데이트"""
        try:
            # 피드백 정보를 메타데이터로 저장
            metadata = {
                'enhancement_id': feedback.enhancement_id,
                'feedback_type': feedback.feedback_type.value,
                'feedback_score': adjusted_score,
                'user_rating': feedback.user_rating,
                'code_accepted': feedback.code_accepted,
                'execution_success': feedback.execution_success,
                'created_at': feedback.created_at.isoformat()
            }
            
            # 벡터 DB에 피드백 정보 저장
            await self.vector_service.add_content(
                content=f"Feedback for {feedback.enhancement_id}: {feedback.feedback_type.value}",
                project_id=feedback.project_id,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error updating vector DB: {e}")
    
    def _generate_recommendation(self, feedback: UserFeedback, feedback_impact: float) -> str:
        """개선 권장사항 생성"""
        if feedback_impact > 0.5:
            return "Excellent feedback! This prompt pattern should be prioritized for similar contexts."
        elif feedback_impact > 0.0:
            return "Positive feedback. Consider reinforcing this prompt pattern."
        elif feedback_impact > -0.5:
            return "Mixed feedback. Review and refine this prompt pattern."
        else:
            return "Negative feedback. This prompt pattern needs significant improvement."
    
    @handle_errors()
    async def get_feedback_statistics(self, project_id: str) -> Dict:
        """피드백 통계 조회"""
        feedbacks = self.feedback_history.get(project_id, [])
        
        if not feedbacks:
            return {
                'total_feedbacks': 0,
                'feedback_distribution': {},
                'average_rating': 0.0,
                'success_rate': 0.0
            }
        
        # 피드백 분포
        feedback_counts = defaultdict(int)
        ratings = []
        success_count = 0
        
        for feedback in feedbacks:
            feedback_counts[feedback.feedback_type.value] += 1
            if feedback.user_rating:
                ratings.append(feedback.user_rating)
            if feedback.code_accepted:
                success_count += 1
        
        return {
            'total_feedbacks': len(feedbacks),
            'feedback_distribution': dict(feedback_counts),
            'average_rating': np.mean(ratings) if ratings else 0.0,
            'success_rate': success_count / len(feedbacks) if feedbacks else 0.0,
            'last_updated': datetime.now().isoformat()
        }
    
    @handle_errors()
    async def analyze_feedback_patterns(self, project_id: str) -> Dict:
        """피드백 패턴 분석"""
        feedbacks = self.feedback_history.get(project_id, [])
        
        if len(feedbacks) < 5:
            return {'message': 'Insufficient feedback data for pattern analysis'}
        
        # 시간별 피드백 트렌드
        time_trends = defaultdict(list)
        for feedback in feedbacks:
            date_key = feedback.created_at.strftime('%Y-%m-%d')
            time_trends[date_key].append(feedback)
        
        # 프롬프트 타입별 성공률
        success_by_type = defaultdict(lambda: {'total': 0, 'success': 0})
        
        for feedback in feedbacks:
            # 프롬프트 타입을 키워드로 추출 (간단한 예)
            prompt_type = self._extract_prompt_type(feedback.original_prompt)
            success_by_type[prompt_type]['total'] += 1
            if feedback.code_accepted:
                success_by_type[prompt_type]['success'] += 1
        
        # 성공률 계산
        success_rates = {}
        for prompt_type, stats in success_by_type.items():
            success_rates[prompt_type] = (
                stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
            )
        
        return {
            'time_trends': {
                date: len(fbs) for date, fbs in time_trends.items()
            },
            'success_rates_by_type': success_rates,
            'total_patterns_analyzed': len(feedbacks),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _extract_prompt_type(self, prompt: str) -> str:
        """프롬프트에서 타입 추출 (간단한 키워드 기반)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['test', 'unit test', 'testing']):
            return 'testing'
        elif any(word in prompt_lower for word in ['refactor', 'improve', 'optimize']):
            return 'refactoring'
        elif any(word in prompt_lower for word in ['create', 'implement', 'build']):
            return 'implementation'
        elif any(word in prompt_lower for word in ['fix', 'debug', 'error']):
            return 'debugging'
        elif any(word in prompt_lower for word in ['api', 'endpoint', 'service']):
            return 'api_development'
        else:
            return 'general'

class FeedbackService:
    """피드백 서비스 메인 클래스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.processor = FeedbackProcessor(vector_service)
        
    async def submit_feedback(self, feedback: UserFeedback) -> FeedbackAnalysis:
        """피드백 제출"""
        return await self.processor.process_feedback(feedback)
    
    async def get_project_feedback_stats(self, project_id: str) -> Dict:
        """프로젝트별 피드백 통계"""
        return await self.processor.get_feedback_statistics(project_id)
    
    async def analyze_project_patterns(self, project_id: str) -> Dict:
        """프로젝트 피드백 패턴 분석"""
        return await self.processor.analyze_feedback_patterns(project_id)
    
    async def get_recommendations_for_prompt(self, prompt: str, project_id: str) -> List[str]:
        """프롬프트에 대한 추천사항"""
        try:
            # 유사한 프롬프트의 피드백 조회
            similar_results = await self.vector_service.search_similar_content(
                query=prompt,
                project_id=project_id,
                limit=5
            )
            
            recommendations = []
            for result in similar_results:
                if result.get('feedback_score', 0) > 0.5:
                    recommendations.append(
                        f"Similar successful pattern: {result.get('content', '')[:100]}..."
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return [] 