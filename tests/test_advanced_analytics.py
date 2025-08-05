#!/usr/bin/env python3
"""
고급 분석 기능 테스트 스크립트

scikit-learn을 활용한 새로운 분석 기능들을 테스트합니다:
- 프롬프트 클러스터링
- TF-IDF 키워드 추출  
- 트렌드 분석
- 고급 유사도 계산
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.vector_service import VectorService
from services.advanced_analytics import AdvancedAnalyticsService
from services.prompt_enhancement_service import PromptEnhancementService
from models.prompt_models import PromptHistory, PromptType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_analytics():
    """고급 분석 기능 테스트"""
    
    print("🧪 고급 분석 기능 테스트 시작")
    print("=" * 50)
    
    # 환경변수 설정 (Docker ChromaDB 사용)
    os.environ["CHROMA_DB_HOST"] = "localhost"
    os.environ["CHROMA_DB_PORT"] = "8001"
    
    # 서비스 초기화
    print("\n📚 서비스 초기화 중...")
    vector_service = VectorService()
    analytics_service = AdvancedAnalyticsService()
    prompt_service = PromptEnhancementService(vector_service)
    
    # 1. 테스트 데이터 생성
    print("\n📝 테스트 프롬프트 데이터 생성 중...")
    test_prompts = [
        "Python으로 FastAPI 서버를 만들어주세요",
        "React 컴포넌트를 TypeScript로 구현해주세요",
        "데이터베이스 스키마 설계를 도와주세요",
        "JWT 인증 시스템을 구현하고 싶어요",
        "REST API 엔드포인트를 만들어주세요",
        "프론트엔드 상태 관리를 어떻게 해야 할까요?",
        "Docker 컨테이너화 하는 방법을 알려주세요",
        "CI/CD 파이프라인 구성을 도와주세요",
        "보안 취약점을 점검하는 방법은?",
        "성능 최적화 방안을 제안해주세요"
    ]
    
    project_id = "test-advanced-analytics"
    
    # 프롬프트 저장
    stored_prompts = []
    for i, prompt_text in enumerate(test_prompts):
        prompt_history = PromptHistory(
            id=f"test-prompt-{i}",
            content=prompt_text,
            project_id=project_id,
            prompt_type=PromptType.USER_QUERY,
            created_at=datetime.now() - timedelta(days=i)
        )
        
        success = await vector_service.store_prompt_history(prompt_history)
        if success:
            stored_prompts.append(prompt_text)
            print(f"   ✅ 저장됨: {prompt_text[:30]}...")
        else:
            print(f"   ❌ 실패: {prompt_text[:30]}...")
    
    print(f"\n📊 총 {len(stored_prompts)}개 프롬프트 저장 완료")
    
    # 2. 유사도 계산 테스트
    print("\n🔍 고급 유사도 계산 테스트")
    try:
        # 두 프롬프트의 임베딩 생성
        emb1 = await vector_service._generate_embedding("Python FastAPI 서버 구현")
        emb2 = await vector_service._generate_embedding("Python으로 웹 서버 만들기")
        
        # 고급 유사도 계산
        similarity = await analytics_service.calculate_advanced_similarity(emb1, emb2)
        print(f"   📈 코사인 유사도: {similarity:.4f}")
        
    except Exception as e:
        print(f"   ❌ 유사도 계산 실패: {e}")
    
    # 3. 프롬프트 클러스터링 테스트
    print("\n🗂️ 프롬프트 클러스터링 테스트")
    try:
        # 저장된 프롬프트들 검색
        prompts = await vector_service.search_similar_prompts(
            query="",
            project_id=project_id,
            limit=20
        )
        
        if len(prompts) >= 3:
            # 임베딩과 텍스트 추출
            embeddings = []
            texts = []
            for prompt in prompts:
                # 실제 임베딩 생성
                embedding = await vector_service._generate_embedding(prompt.get('content', ''))
                embeddings.append(embedding)
                texts.append(prompt.get('content', ''))
            
            # 클러스터링 수행
            clustering_result = await analytics_service.cluster_prompts(
                prompt_embeddings=embeddings,
                prompt_texts=texts,
                n_clusters=3
            )
            
            print(f"   📊 클러스터 개수: {len(clustering_result.get('clusters', []))}")
            print(f"   📈 실루엣 점수: {clustering_result.get('silhouette_score', 0):.4f}")
            
            for i, cluster in enumerate(clustering_result.get('clusters', [])):
                print(f"   🗂️ 클러스터 {i+1}: {cluster.get('size', 0)}개 프롬프트")
                print(f"      대표: {cluster.get('representative_prompt', '')[:50]}...")
        else:
            print(f"   ⚠️ 클러스터링을 위한 충분한 데이터가 없습니다. ({len(prompts)}개)")
            
    except Exception as e:
        print(f"   ❌ 클러스터링 실패: {e}")
    
    # 4. TF-IDF 키워드 추출 테스트
    print("\n🔍 TF-IDF 키워드 추출 테스트")
    try:
        features_result = await analytics_service.extract_text_features(test_prompts)
        
        print(f"   📚 어휘 크기: {features_result.get('vocabulary_size', 0)}")
        
        top_features = features_result.get('top_features', [])
        if top_features:
            print("   🏆 상위 키워드:")
            for feature in top_features[:10]:
                print(f"      - {feature.get('term', '')}: {feature.get('score', 0):.4f}")
        else:
            print("   ⚠️ 추출된 키워드가 없습니다.")
            
    except Exception as e:
        print(f"   ❌ 키워드 추출 실패: {e}")
    
    # 5. 트렌드 분석 테스트
    print("\n📈 트렌드 분석 테스트")
    try:
        # 메타데이터 포함한 프롬프트 데이터 구성
        prompt_data = []
        for i, prompt_text in enumerate(test_prompts):
            prompt_data.append({
                "content": prompt_text,
                "created_at": datetime.now() - timedelta(days=i),
                "prompt_type": "user_query"
            })
        
        trends = await analytics_service.analyze_prompt_trends(prompt_data)
        
        print(f"   📊 분석된 프롬프트 수: {trends.get('total_prompts', 0)}")
        
        # 시간 패턴
        temporal = trends.get('temporal_patterns', {})
        if temporal:
            print(f"   🕐 피크 시간: {temporal.get('peak_hour', 'N/A')}시")
            print(f"   📅 활동 시간대: {temporal.get('total_hours', 0)}시간")
        
        # 길이 분포
        length = trends.get('length_distribution', {})
        if length:
            print(f"   📏 평균 길이: {length.get('average_length', 0):.1f}자")
            print(f"   📐 최대 길이: {length.get('max_length', 0)}자")
        
        # 복잡도
        complexity = trends.get('complexity_metrics', {})
        if complexity:
            print(f"   🧮 평균 단어 수: {complexity.get('average_word_count', 0):.1f}")
            print(f"   📝 평균 문장 수: {complexity.get('average_sentence_count', 0):.1f}")
            
    except Exception as e:
        print(f"   ❌ 트렌드 분석 실패: {e}")
    
    # 6. 향상된 프롬프트 개선 테스트
    print("\n🚀 향상된 프롬프트 개선 테스트")
    try:
        from models.prompt_models import PromptEnhanceRequest
        
        request = PromptEnhanceRequest(
            original_prompt="Python 웹 서버 만들어줘",
            project_id=project_id,
            context_limit=5
        )
        
        result = await prompt_service.enhance_prompt(request)
        
        print(f"   📊 신뢰도 점수: {result.confidence_score:.4f}")
        print(f"   💡 제안사항 수: {len(result.suggestions)}")
        print(f"   🔗 사용된 컨텍스트: {len(result.context_used)}개")
        
        if result.suggestions:
            print("   💡 주요 제안사항:")
            for suggestion in result.suggestions[:3]:
                print(f"      - {suggestion}")
                
    except Exception as e:
        print(f"   ❌ 프롬프트 개선 실패: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 고급 분석 기능 테스트 완료!")
    print("\n📋 사용된 scikit-learn 기능들:")
    print("   - 코사인 유사도 (cosine_similarity)")
    print("   - K-평균 클러스터링 (KMeans)")
    print("   - TF-IDF 벡터화 (TfidfVectorizer)")
    print("   - 실루엣 분석 (silhouette_score)")
    print("   - 차원 축소 (PCA, t-SNE) - 향후 시각화용")

if __name__ == "__main__":
    asyncio.run(test_advanced_analytics()) 