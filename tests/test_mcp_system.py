#!/usr/bin/env python3
"""
MCP 서버 테스트 스크립트
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_system():
    """MCP 시스템 테스트"""
    try:
        # 서비스 import
        from services.vector_service import VectorService
        from services.prompt_enhancement_service import PromptEnhancementService
        from services.file_indexing_service import FileIndexingService
        from models.prompt_models import PromptEnhanceRequest
        
        logger.info("🚀 MCP 시스템 테스트 시작")
        
        # 1. 서비스 초기화
        logger.info("1. 서비스 초기화 중...")
        vector_service = VectorService()
        enhancement_service = PromptEnhancementService(vector_service)
        file_indexing_service = FileIndexingService(vector_service)
        
        # 2. 현재 프로젝트 인덱싱 테스트
        logger.info("2. 현재 프로젝트 파일 인덱싱 테스트...")
        current_project_path = os.getcwd()
        project_id = "mcp-server-test"
        
        result = await file_indexing_service.index_project_files(
            current_project_path, 
            project_id
        )
        
        if result["success"]:
            logger.info(f"✅ 인덱싱 성공: {result['indexed_files_count']}개 파일")
            logger.info(f"   프로젝트명: {result['project_name']}")
            logger.info(f"   기술 스택: {result['tech_stack']}")
            logger.info(f"   파일 패턴: {result['file_patterns']}")
        else:
            logger.error(f"❌ 인덱싱 실패: {result.get('error', '알 수 없는 오류')}")
            return
        
        # 3. 프롬프트 개선 테스트
        logger.info("3. 프롬프트 개선 테스트...")
        
        test_prompts = [
            "Python FastAPI 서버에서 비동기 API 엔드포인트를 만들어줘",
            "벡터 데이터베이스 검색 기능을 구현하고 싶어",
            "MCP 서버에 새로운 도구를 추가하는 방법을 알려줘",
            "프롬프트 히스토리를 저장하는 함수를 개선해줘"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"3.{i} 테스트 프롬프트: {prompt}")
            
            request = PromptEnhanceRequest(
                original_prompt=prompt,
                project_id=project_id,
                context_limit=3
            )
            
            enhanced_result = await enhancement_service.enhance_prompt(request)
            
            logger.info(f"   신뢰도 점수: {enhanced_result.confidence_score:.2f}")
            logger.info(f"   사용된 컨텍스트: {len(enhanced_result.context_used)}개")
            logger.info(f"   제안사항: {len(enhanced_result.suggestions)}개")
            
            if enhanced_result.enhanced_prompt != request.original_prompt:
                logger.info("   ✅ 프롬프트가 개선되었습니다")
            else:
                logger.info("   ⚠️ 프롬프트 개선이 제한적입니다")
            
            # 프롬프트 히스토리에 저장
            from models.prompt_models import PromptHistory, PromptType
            
            await vector_service.store_prompt_history(
                PromptHistory(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    content=prompt,
                    prompt_type=PromptType.USER_QUERY,
                    metadata={"test": True},
                    created_at=datetime.now()
                )
            )
        
        # 4. 파일 검색 테스트
        logger.info("4. 파일 검색 테스트...")
        
        search_queries = [
            "FastAPI 서버 설정",
            "벡터 서비스 구현",
            "프롬프트 개선 로직",
            "MCP 도구 정의"
        ]
        
        for query in search_queries:
            results = await vector_service.search_similar_prompts(
                query=query,
                project_id=project_id,
                limit=3
            )
            
            # 파일 내용만 필터링
            file_results = [
                r for r in results 
                if r.get('metadata', {}).get('is_file_content', False)
            ]
            
            logger.info(f"   검색어 '{query}': {len(file_results)}개 파일 결과")
        
        # 5. 프로젝트 컨텍스트 조회 테스트
        logger.info("5. 프로젝트 컨텍스트 조회 테스트...")
        context = await vector_service.get_project_context(project_id)
        
        if context:
            logger.info("   ✅ 프로젝트 컨텍스트 조회 성공")
            metadata = context.get("metadata", {})
            logger.info(f"   프로젝트명: {metadata.get('project_name', 'N/A')}")
            logger.info(f"   설명: {metadata.get('description', 'N/A')[:100]}...")
        else:
            logger.info("   ⚠️ 프로젝트 컨텍스트를 찾을 수 없습니다")
        
        logger.info("🎉 모든 테스트가 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """메인 함수"""
    print("=" * 60)
    print("🔧 MCP 프롬프트 향상 시스템 테스트")
    print("=" * 60)
    
    await test_mcp_system()
    
    print("\n" + "=" * 60)
    print("📚 사용 방법:")
    print("1. 다른 프로젝트를 인덱싱: index_project_files('/path/to/project', 'project_id')")
    print("2. 프롬프트 개선: enhance_prompt('your prompt', 'project_id')")
    print("3. 대화 저장: store_conversation('user_prompt', 'ai_response', 'project_id')")
    print("4. 파일 검색: search_project_files('query', 'project_id')")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 