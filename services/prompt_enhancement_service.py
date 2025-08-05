import logging
import os
from typing import List, Dict, Any, Optional
from models.prompt_models import PromptEnhanceRequest, PromptEnhanceResponse
from services.vector_service import VectorService, DeepSeekLLM
from services.advanced_analytics import AdvancedAnalyticsService
from config import settings

# 선택적 임포트
try:
    from langchain_openai import OpenAI
except ImportError:
    try:
        from langchain.llms import OpenAI
    except ImportError:
        OpenAI = None

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    PromptTemplate = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        HumanMessage = None
        SystemMessage = None

logger = logging.getLogger(__name__)

class PromptEnhancementService:
    """프롬프트 개선 서비스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.analytics_service = AdvancedAnalyticsService()
        self.llm = None
        
        # LLM 초기화 (설정에 따라 OpenAI 또는 DeepSeek)
        embedding_model_type = os.getenv("EMBEDDING_MODEL_TYPE", settings.embedding_model_type)
        
        if embedding_model_type == "openai" and settings.openai_api_key and OpenAI:
            try:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    temperature=0.7,
                    max_tokens=2000
                )
            except Exception as e:
                logger.warning(f"OpenAI LLM 초기화 실패: {e}")
        elif embedding_model_type == "deepseek":
            try:
                deepseek_api_base = os.getenv("DEEPSEEK_API_BASE", settings.deepseek_api_base)
                deepseek_llm_model = os.getenv("DEEPSEEK_LLM_MODEL", settings.deepseek_llm_model)
                self.llm = DeepSeekLLM(
                    api_base=deepseek_api_base,
                    model_name=deepseek_llm_model,
                    temperature=0.7,
                    max_tokens=2000
                )
                logger.info(f"DeepSeek LLM 초기화 완료: {deepseek_api_base} (모델: {deepseek_llm_model})")
            except Exception as e:
                logger.warning(f"DeepSeek LLM 초기화 실패: {e}")
        else:
            logger.info("LLM을 사용할 수 없습니다. 기본 프롬프트 포맷터를 사용합니다.")
        
        # 개선된 프롬프트 템플릿 정의
        self.enhancement_template = None
        self.enhancement_chain = None
        self.prompt_formatter = StandardPromptFormatter()
        
        if PromptTemplate and self.llm:
            try:
                self.enhancement_template = PromptTemplate(
                    input_variables=["original_prompt", "project_context", "similar_prompts", "tech_stack", "file_context"],
                    template=self.prompt_formatter.get_enhancement_template()
                )
                
                # LangChain 새로운 방식: prompt | llm (OpenAI와 DeepSeek 모두 지원)
                try:
                    self.enhancement_chain = self.enhancement_template | self.llm
                    logger.info(f"프롬프트 개선 체인 생성 성공: {type(self.llm).__name__}")
                except Exception as e:
                    logger.warning(f"프롬프트 개선 체인 생성 실패: {e}")
                    self.enhancement_chain = None
            except Exception as e:
                logger.warning(f"LangChain 초기화 실패: {e}")
    
    async def enhance_prompt(self, request: PromptEnhanceRequest) -> PromptEnhanceResponse:
        """프롬프트 개선"""
        try:
            # 1. 프로젝트 컨텍스트 조회
            project_context = await self.vector_service.get_project_context(request.project_id)
            
            # 2. 유사한 프롬프트 검색
            context_limit = request.context_limit or settings.max_context_length
            similar_prompts = await self.vector_service.search_similar_prompts(
                query=request.original_prompt,
                project_id=request.project_id,
                limit=context_limit
            )
            
            # 유사도 임계값 필터링
            similar_prompts = [
                prompt for prompt in similar_prompts 
                if prompt.get('similarity', 0) >= settings.similarity_threshold
            ]
            
            # 3. 관련 파일 컨텍스트 검색 (프롬프트와 관련된 코드/문서)
            file_context = await self.vector_service.search_similar_prompts(
                query=request.original_prompt + " code implementation documentation",
                project_id=request.project_id,
                limit=3
            )
            
            # 파일 컨텍스트만 필터링
            file_context = [
                item for item in file_context 
                if item.get('metadata', {}).get('is_file_content', False)
            ]
            
            # 4. 표준화된 프롬프트 포맷으로 개선
            enhanced_result = await self._enhance_with_standard_format(
                request.original_prompt,
                project_context,
                similar_prompts,
                file_context
            )
            
            # 5. 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score(
                project_context, similar_prompts, file_context, request.original_prompt
            )
            
            # 6. 컨텍스트 사용 정보 구성
            context_used = []
            # 프롬프트 히스토리 추가
            for prompt in similar_prompts:
                context_used.append(f"프롬프트: {prompt['content'][:100]}...")
            # 파일 컨텍스트 추가
            for file_item in file_context:
                file_path = file_item.get('metadata', {}).get('file_path', 'Unknown')
                context_used.append(f"파일: {file_path}")
            
            # 7. 개선 제안사항 생성
            suggestions = self._generate_suggestions(project_context, similar_prompts, file_context)
            
            return PromptEnhanceResponse(
                enhanced_prompt=enhanced_result.strip(),
                confidence_score=confidence_score,
                context_used=context_used,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"프롬프트 개선 실패: {e}")
            # 실패 시 원본 프롬프트 반환
            return PromptEnhanceResponse(
                enhanced_prompt=request.original_prompt,
                confidence_score=0.0,
                context_used=[],
                suggestions=["프롬프트 개선 중 오류가 발생했습니다."]
            )
    
    async def _enhance_with_standard_format(
        self,
        original_prompt: str,
        project_context: Optional[Dict[str, Any]],
        similar_prompts: List[Dict[str, Any]],
        file_context: List[Dict[str, Any]]
    ) -> str:
        """표준화된 포맷으로 프롬프트 개선"""
        
        # 컨텍스트 정보 구성
        context_text = self._format_context(project_context)
        tech_stack_text = self._format_tech_stack(project_context)
        similar_prompts_text = self._format_similar_prompts(similar_prompts)
        file_context_text = self._format_file_context(file_context)
        
        # LLM을 통한 프롬프트 개선
        if self.enhancement_chain:
            try:
                enhanced_result = await self.enhancement_chain.ainvoke({
                    "original_prompt": original_prompt,
                    "project_context": context_text,
                    "similar_prompts": similar_prompts_text,
                    "tech_stack": tech_stack_text,
                    "file_context": file_context_text
                })
                
                # LangChain 새로운 방식에서는 AIMessage 객체나 문자열을 반환할 수 있음
                if hasattr(enhanced_result, 'content'):
                    enhanced_result = enhanced_result.content
                elif not isinstance(enhanced_result, str):
                    enhanced_result = str(enhanced_result)
                    
                return enhanced_result
            except Exception as e:
                logger.warning(f"LLM 프롬프트 개선 실패: {e}")
        elif self.llm and self.enhancement_template:
            try:
                # DeepSeek LLM 직접 호출
                prompt_text = self.enhancement_template.format(
                    original_prompt=original_prompt,
                    project_context=context_text,
                    similar_prompts=similar_prompts_text,
                    tech_stack=tech_stack_text,
                    file_context=file_context_text
                )
                enhanced_result = await self.llm.arun(prompt=prompt_text)
                return enhanced_result
            except Exception as e:
                logger.warning(f"DeepSeek LLM 프롬프트 개선 실패: {e}")
        
        # LLM이 실패하면 표준 포맷터 사용
        return self.prompt_formatter.format_enhanced_prompt(
            original_prompt=original_prompt,
            project_context=project_context,
            similar_prompts=similar_prompts,
            file_context=file_context
        )
    
    def _format_context(self, project_context: Optional[Dict[str, Any]]) -> str:
        """프로젝트 컨텍스트 포맷팅"""
        if not project_context:
            return "프로젝트 컨텍스트가 없습니다."
        
        metadata = project_context.get("metadata", {})
        return f"""
프로젝트명: {metadata.get('project_name', 'Unknown')}
설명: {metadata.get('description', 'No description')}
"""

    def _format_tech_stack(self, project_context: Optional[Dict[str, Any]]) -> str:
        """기술 스택 포맷팅"""
        if not project_context:
            return "기술 스택 정보가 없습니다."
        
        metadata = project_context.get("metadata", {})
        tech_stack = metadata.get('tech_stack', '').split(',')
        return f"기술 스택: {', '.join(tech_stack)}" if tech_stack else "기술 스택 정보 없음"

    def _format_similar_prompts(self, similar_prompts: List[Dict[str, Any]]) -> str:
        """유사한 프롬프트들 포맷팅"""
        if not similar_prompts:
            return "유사한 프롬프트가 없습니다."
        
        formatted = []
        for i, prompt in enumerate(similar_prompts[:3], 1):
            similarity = prompt.get('similarity', 0)
            content = prompt.get('content', '')[:200] + "..."
            formatted.append(f"{i}. (유사도: {similarity:.2f}) {content}")
        
        return "\n".join(formatted)

    def _format_file_context(self, file_context: List[Dict[str, Any]]) -> str:
        """파일 컨텍스트 포맷팅"""
        if not file_context:
            return "관련 파일 정보가 없습니다."
        
        formatted = []
        for item in file_context:
            file_path = item.get('metadata', {}).get('file_path', 'Unknown')
            content = item.get('content', '')[:300] + "..."
            formatted.append(f"파일: {file_path}\n내용: {content}")
        
        return "\n\n".join(formatted)

    def _calculate_confidence_score(
        self, 
        project_context: Optional[Dict[str, Any]], 
        similar_prompts: List[Dict[str, Any]], 
        file_context: List[Dict[str, Any]],
        original_prompt: str
    ) -> float:
        """신뢰도 점수 계산"""
        score = 0.0
        
        # 기본 점수 (프로젝트 컨텍스트 존재)
        if project_context:
            score += 0.3
        
        # 유사 프롬프트 점수
        if similar_prompts:
            avg_similarity = sum(p.get('similarity', 0) for p in similar_prompts) / len(similar_prompts)
            score += avg_similarity * 0.4
        
        # 파일 컨텍스트 점수
        if file_context:
            score += 0.2
        
        # 고급 분석 점수
        advanced_score = self._calculate_advanced_score(original_prompt, similar_prompts, file_context)
        score += advanced_score * 0.1
        
        return min(score, 1.0)

    def _calculate_advanced_score(
        self,
        original_prompt: str,
        similar_prompts: List[Dict[str, Any]],
        file_context: List[Dict[str, Any]]
    ) -> float:
        """고급 분석 점수 계산"""
        try:
            # 프롬프트 복잡도 분석
            complexity_score = self._analyze_prompt_complexity_score(original_prompt)
            
            # 컨텍스트 관련성 분석
            relevance_score = self._calculate_context_relevance(original_prompt, similar_prompts, file_context)
            
            # 텍스트 품질 분석
            quality_score = self._calculate_text_quality(original_prompt)
            
            return (complexity_score + relevance_score + quality_score) / 3
            
        except Exception as e:
            logger.warning(f"고급 분석 점수 계산 실패: {e}")
            return 0.5

    def _analyze_prompt_complexity_score(self, prompt: str) -> float:
        """프롬프트 복잡도 분석"""
        # 단어 수, 문장 수, 기술 용어 등을 고려한 복잡도 점수
        words = len(prompt.split())
        sentences = len([s for s in prompt.split('.') if s.strip()])
        
        # 기술 용어 카운트
        tech_terms = ['function', 'class', 'method', 'variable', 'API', 'database', 'server', 'client']
        tech_count = sum(1 for term in tech_terms if term.lower() in prompt.lower())
        
        complexity = min((words / 50) + (sentences / 10) + (tech_count / 5), 1.0)
        return complexity

    def _calculate_context_relevance(
        self,
        prompt: str,
        similar_prompts: List[Dict[str, Any]],
        file_context: List[Dict[str, Any]]
    ) -> float:
        """컨텍스트 관련성 계산"""
        if not similar_prompts and not file_context:
            return 0.0
        
        # 유사 프롬프트 관련성
        prompt_relevance = 0.0
        if similar_prompts:
            prompt_relevance = sum(p.get('similarity', 0) for p in similar_prompts) / len(similar_prompts)
        
        # 파일 컨텍스트 관련성 (간단한 키워드 매칭)
        file_relevance = 0.0
        if file_context:
            prompt_words = set(prompt.lower().split())
            file_matches = 0
            for item in file_context:
                content = item.get('content', '').lower()
                file_words = set(content.split())
                overlap = len(prompt_words.intersection(file_words))
                file_matches += min(overlap / max(len(prompt_words), 1), 1.0)
            file_relevance = file_matches / len(file_context)
        
        return (prompt_relevance + file_relevance) / 2

    def _calculate_text_quality(self, prompt: str) -> float:
        """텍스트 품질 계산"""
        # 기본적인 텍스트 품질 지표
        if not prompt.strip():
            return 0.0
        
        # 길이 적절성
        length_score = min(len(prompt) / 100, 1.0) if len(prompt) > 10 else 0.3
        
        # 문장 구조 (대소문자, 구두점 등)
        structure_score = 0.5
        if prompt[0].isupper():
            structure_score += 0.2
        if any(punct in prompt for punct in '.!?'):
            structure_score += 0.2
        if len(prompt.split()) > 3:
            structure_score += 0.1
        
        return min((length_score + structure_score) / 2, 1.0)

    def _generate_suggestions(
        self, 
        project_context: Optional[Dict[str, Any]], 
        similar_prompts: List[Dict[str, Any]],
        file_context: List[Dict[str, Any]]
    ) -> List[str]:
        """개선 제안사항 생성"""
        suggestions = []
        
        # 프로젝트 컨텍스트 기반 제안
        if project_context:
            metadata = project_context.get("metadata", {})
            tech_stack = metadata.get("tech_stack", "").split(",")
            if tech_stack:
                suggestions.append(f"프로젝트의 기술 스택({', '.join(tech_stack)})을 고려하여 더 구체적인 요구사항을 명시하세요.")
        
        # 유사 프롬프트 기반 제안
        if similar_prompts:
            suggestions.append("과거 유사한 프롬프트들의 패턴을 참고하여 더 명확한 표현을 사용하세요.")
        
        # 파일 컨텍스트 기반 제안
        if file_context:
            suggestions.append("관련 코드베이스의 구조와 패턴을 고려하여 기존 아키텍처와 일관성을 유지하세요.")
        
        # 일반적인 제안
        suggestions.extend([
            "구체적인 입력/출력 예시를 포함하여 요구사항을 명확히 하세요.",
            "예상되는 에러 상황과 처리 방법을 명시하세요.",
            "성능 요구사항이나 제약사항이 있다면 구체적으로 기술하세요."
        ])
        
        return suggestions[:5]  # 최대 5개 제안

    def _simple_prompt_enhancement(self, original_prompt: str, context: str, tech_stack: str, similar_prompts: str, file_context: str) -> str:
        """간단한 프롬프트 개선 (LLM 없이)"""
        return self.prompt_formatter.format_enhanced_prompt(
            original_prompt=original_prompt,
            project_context={"metadata": {"description": context}},
            similar_prompts=[{"content": similar_prompts}],
            file_context=[{"content": file_context}]
        )

    async def analyze_prompt_patterns(self, project_id: str) -> Dict[str, Any]:
        """프롬프트 패턴 분석"""
        try:
            return await self.analytics_service.analyze_prompt_patterns(project_id)
        except Exception as e:
            logger.error(f"프롬프트 패턴 분석 실패: {e}")
            return {"error": str(e)}


class StandardPromptFormatter:
    """표준화된 프롬프트 포맷터"""
    
    def get_enhancement_template(self) -> str:
        """개선용 템플릿 반환"""
        return """
# 🚀 AI 프롬프트 개선 시스템

## 📋 분석 컨텍스트
**프로젝트 정보:**
{project_context}

**기술 스택:**
{tech_stack}

**관련 코드/문서:**
{file_context}

**유사한 과거 프롬프트:**
{similar_prompts}

## 🎯 개선 대상 프롬프트
```
{original_prompt}
```

## 💡 개선 지침
1. **명확성**: 프로젝트 컨텍스트를 활용하여 더 구체적으로 작성
2. **일관성**: 기존 코드베이스와 아키텍처 패턴 준수
3. **완전성**: 필요한 정보와 제약사항 모두 포함
4. **실용성**: 실제 구현 가능한 명확한 지침 제공

## ✨ 개선된 프롬프트
"""
    
    def format_enhanced_prompt(
        self,
        original_prompt: str,
        project_context: Optional[Dict[str, Any]] = None,
        similar_prompts: Optional[List[Dict[str, Any]]] = None,
        file_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """표준 포맷으로 프롬프트 개선"""
        
        # 프로젝트 컨텍스트 추출
        project_name = "Unknown Project"
        tech_stack = []
        if project_context:
            metadata = project_context.get("metadata", {})
            project_name = metadata.get("project_name", "Unknown Project")
            tech_stack = metadata.get("tech_stack", "").split(",")
        
        # 개선된 프롬프트 구성
        enhanced_prompt = f"""
# 🎯 개선된 프롬프트

## 📋 프로젝트 컨텍스트
- **프로젝트**: {project_name}
- **기술 스택**: {", ".join(tech_stack) if tech_stack else "미지정"}

## 🚀 요구사항
{original_prompt}

## 🔧 구현 지침
1. **아키텍처**: 기존 프로젝트 구조와 일관성 유지
2. **코드 품질**: 클린 코드 원칙 준수
3. **테스트**: 적절한 테스트 코드 포함
4. **문서화**: 필요한 주석과 문서 작성

## 📝 추가 고려사항
- 에러 처리 및 예외 상황 대응
- 성능 최적화 방안
- 보안 및 검증 로직
- 유지보수성 확보
"""
        
        # 유사 프롬프트 패턴 적용
        if similar_prompts:
            enhanced_prompt += "\n\n## 📚 참고 패턴\n"
            for i, prompt in enumerate(similar_prompts[:2], 1):
                content = prompt.get("content", "")[:150] + "..."
                enhanced_prompt += f"{i}. {content}\n"
        
        # 파일 컨텍스트 적용
        if file_context:
            enhanced_prompt += "\n\n## 🔍 관련 코드베이스\n"
            for item in file_context[:2]:
                file_path = item.get("metadata", {}).get("file_path", "Unknown")
                enhanced_prompt += f"- 참고 파일: {file_path}\n"
        
        return enhanced_prompt.strip() 