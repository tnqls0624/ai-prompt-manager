#!/usr/bin/env python3
"""
FastMCP 서버 테스트 스크립트
"""

import asyncio
import logging
import sys
from mcp_server import (
    enhance_prompt,
    store_prompt,
    store_conversation,
    search_similar_prompts,
    analyze_conversation_patterns,
    get_server_status,
    initialize_services
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-fastmcp")

async def test_fastmcp_server():
    """FastMCP 서버 기능 테스트"""
    
    print("=== FastMCP 서버 테스트 시작 ===\n")
    
    try:
        # 1. 서비스 초기화 테스트
        print("1. 서비스 초기화 테스트...")
        await initialize_services()
        print("✅ 서비스 초기화 성공\n")
        
        # 2. 서버 상태 확인
        print("2. 서버 상태 확인...")
        status = await get_server_status()
        print(f"✅ 서버 상태: {status.get('status')}")
        print(f"   이름: {status.get('server_name')}")
        print(f"   버전: {status.get('version')}")
        print(f"   전송 방식: {status.get('transport')}")
        print(f"   서비스 상태: {status.get('services')}")
        print("")
        
        # 3. 프롬프트 개선 테스트
        print("3. 프롬프트 개선 테스트...")
        enhance_result = await enhance_prompt(
            prompt="React 컴포넌트를 만들어주세요",
            project_id="test-project",
            context_limit=3
        )
        print(f"✅ 프롬프트 개선 결과:")
        print(f"   성공: {enhance_result.get('success')}")
        if enhance_result.get('success'):
            enhanced = enhance_result.get('enhanced_prompt', '')
            print(f"   개선된 프롬프트: {enhanced[:100]}...")
        else:
            print(f"   오류: {enhance_result.get('error')}")
        print("")
        
        # 4. 프롬프트 저장 테스트
        print("4. 프롬프트 저장 테스트...")
        store_result = await store_prompt(
            prompt="테스트 사용자 프롬프트",
            project_id="test-project",
            prompt_type="user_query"
        )
        print(f"✅ 프롬프트 저장 결과:")
        print(f"   성공: {store_result.get('success')}")
        print(f"   메시지: {store_result.get('message')}")
        print(f"   ID: {store_result.get('id')}")
        print("")
        
        # 5. 대화 저장 테스트
        print("5. 대화 저장 테스트...")
        conversation_result = await store_conversation(
            user_prompt="React hooks 사용법을 알려주세요",
            ai_response="React hooks는 함수형 컴포넌트에서 상태 관리를 할 수 있게 해주는 기능입니다.",
            project_id="test-project"
        )
        print(f"✅ 대화 저장 결과:")
        print(f"   성공: {conversation_result.get('success')}")
        print(f"   메시지: {conversation_result.get('message')}")
        print("")
        
        # 6. 유사 프롬프트 검색 테스트
        print("6. 유사 프롬프트 검색 테스트...")
        search_result = await search_similar_prompts(
            query="React 컴포넌트",
            project_id="test-project",
            limit=3
        )
        print(f"✅ 검색 결과:")
        print(f"   성공: {search_result.get('success')}")
        print(f"   결과 개수: {search_result.get('total_results')}")
        if search_result.get('success'):
            results = search_result.get('results', [])
            for i, result in enumerate(results[:3], 1):
                prompt = result.get('prompt', '')
                print(f"   {i}. {prompt[:50]}...")
        print("")
        
        # 7. 패턴 분석 테스트
        print("7. 패턴 분석 테스트...")
        pattern_result = await analyze_conversation_patterns(
            project_id="test-project"
        )
        print(f"✅ 패턴 분석 결과:")
        print(f"   성공: {pattern_result.get('success')}")
        print(f"   메시지: {pattern_result.get('message')}")
        print(f"   제안: {pattern_result.get('suggestion')}")
        print("")
        
        print("🎉 모든 테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        print(f"❌ 테스트 실패: {e}")
        return False
    
    return True

async def main():
    """메인 함수"""
    print("FastMCP 서버 기능 테스트")
    print("=" * 50)
    
    success = await test_fastmcp_server()
    
    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        print("\n다음 명령으로 FastMCP 서버를 시작할 수 있습니다:")
        print("  ./start_fastmcp_server.sh")
        print("\nCursor에서 사용하려면 다음 설정을 추가하세요:")
        print('{')
        print('  "mcpServers": {')
        print('    "prompt-enhancement": {')
        print('      "command": "python",')
        print('      "args": ["mcp_server.py"]')
        print('    }')
        print('  }')
        print('}')
    else:
        print("\n❌ 테스트 중 오류가 발생했습니다.")
        print("로그를 확인하고 문제를 해결해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 