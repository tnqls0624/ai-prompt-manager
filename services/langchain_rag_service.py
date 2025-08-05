"""
LangChain RAG 파이프라인 서비스
DocumentLoader, TextSplitter, Retriever, PromptTemplate, LLMChain 등을 활용한 검색 증강 생성
"""

from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.base import VectorStore
import os
import logging
from datetime import datetime
from config import settings
from services.vector_service import VectorService, DeepSeekLLM
from services.error_handler import handle_errors, ErrorCategory, ErrorLevel
import chromadb
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomChromaRetriever(BaseRetriever):
    """Chroma 벡터 스토어를 위한 커스텀 리트리버"""
    
    def __init__(self, vector_service: VectorService, project_id: str = "default", k: int = 5):
        super().__init__()
        self.vector_service = vector_service
        self.project_id = project_id
        self.k = k
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """쿼리와 관련된 문서들을 검색"""
        try:
            # 벡터 서비스를 통해 유사한 문서들 검색
            results = self.vector_service.search_similar_content(
                query=query,
                project_id=self.project_id,
                limit=self.k
            )
            
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': result.get('source', ''),
                        'score': result.get('score', 0.0),
                        'type': result.get('type', 'unknown'),
                        'timestamp': result.get('timestamp', ''),
                        'file_path': result.get('file_path', '')
                    }
                )
                documents.append(doc)
            
            logger.info(f"검색된 문서 수: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return []

class LangChainRAGService:
    """LangChain 기반 RAG 파이프라인 서비스"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        
        # 임베딩 모델 초기화 - config에 따라 선택
        self.embedding_model_type = os.getenv("EMBEDDING_MODEL_TYPE", settings.embedding_model_type)
        
        if self.embedding_model_type == "openai" and settings.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-3-small"
            )
            self.llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_tokens=4000
            )
        elif self.embedding_model_type == "deepseek":
            # DeepSeek 임베딩과 LLM 모두 사용
            self.embeddings = vector_service.embeddings
            deepseek_api_base = os.getenv("DEEPSEEK_API_BASE", settings.deepseek_api_base)
            deepseek_llm_model = os.getenv("DEEPSEEK_LLM_MODEL", settings.deepseek_llm_model)
            self.llm = DeepSeekLLM(
                api_base=deepseek_api_base,
                model_name=deepseek_llm_model,
                temperature=0.1,
                max_tokens=4000
            )
        else:
            # 다른 모델 타입 또는 설정 부족
            self.embeddings = vector_service.embeddings
            self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # 프롬프트 템플릿 정의
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""# Context from my project
{context}

---
Based on the context above, please write code to satisfy the following requirement.

Requirement: {question}

Please provide:
1. A complete, working code solution
2. Brief explanation of the implementation
3. Any necessary imports or dependencies
4. Usage examples if applicable

Focus on using the project context to maintain consistency with existing code patterns, naming conventions, and architecture."""
        )
        
        # LLM 체인 생성 (OpenAI와 DeepSeek 모두 지원)
        if self.llm:
            try:
                self.llm_chain = LLMChain(
                    llm=self.llm,
                    prompt=self.prompt_template,
                    verbose=True
                )
                logger.info(f"LLM 체인 생성 성공: {type(self.llm).__name__}")
            except Exception as e:
                logger.warning(f"LLM 체인 생성 실패: {e}")
                self.llm_chain = None
        else:
            self.llm_chain = None
        
        logger.info(f"LangChain RAG 서비스 초기화 완료 (임베딩: {self.embedding_model_type}, LLM: {'OpenAI' if self.embedding_model_type == 'openai' else 'DeepSeek' if self.embedding_model_type == 'deepseek' else 'None'})")
    
    @handle_errors(
        category=ErrorCategory.AI_SERVICE,
        level=ErrorLevel.MEDIUM,
        user_message="RAG 파이프라인 실행 중 오류가 발생했습니다."
    )
    async def generate_enhanced_prompt(
        self, 
        user_prompt: str, 
        project_id: str = "default",
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """
        사용자 프롬프트를 RAG 파이프라인으로 향상시켜 컨텍스트가 포함된 프롬프트를 생성
        
        Args:
            user_prompt: 사용자 입력 프롬프트
            project_id: 프로젝트 ID
            context_limit: 검색할 컨텍스트 개수
            
        Returns:
            향상된 프롬프트와 관련 정보
        """
        try:
            # 1. 리트리버 생성
            retriever = CustomChromaRetriever(
                vector_service=self.vector_service,
                project_id=project_id,
                k=context_limit
            )
            
            # 2. 관련 문서 검색
            relevant_docs = retriever.get_relevant_documents(user_prompt)
            
            # 3. 컨텍스트 구성
            context_parts = []
            for doc in relevant_docs:
                context_info = f"## {doc.metadata.get('source', 'Unknown Source')}\n"
                context_info += f"**Type:** {doc.metadata.get('type', 'unknown')}\n"
                context_info += f"**Content:**\n{doc.page_content}\n"
                context_parts.append(context_info)
            
            combined_context = "\n\n".join(context_parts)
            
            # 4. 최종 프롬프트 생성
            enhanced_prompt = self.prompt_template.format(
                context=combined_context,
                question=user_prompt
            )
            
            # 5. 메타데이터 정보 구성
            metadata = {
                "original_prompt": user_prompt,
                "project_id": project_id,
                "context_sources": [doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
                "context_count": len(relevant_docs),
                "enhancement_timestamp": datetime.now().isoformat(),
                "model_info": {
                                    "embeddings_model": "text-embedding-3-small" if self.embedding_model_type == "openai" else f"deepseek-{settings.deepseek_embedding_model}",
                "llm_model": "gpt-4o-mini" if self.embedding_model_type == "openai" else f"deepseek-{settings.deepseek_llm_model}" if self.embedding_model_type == "deepseek" else "none"
                }
            }
            
            logger.info(f"프롬프트 향상 완료: {len(relevant_docs)}개 컨텍스트 포함")
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": user_prompt,
                "context": combined_context,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"RAG 파이프라인 실행 중 오류: {str(e)}")
            return {
                "enhanced_prompt": user_prompt,  # 오류 시 원본 프롬프트 반환
                "original_prompt": user_prompt,
                "context": "",
                "metadata": {
                    "error": str(e),
                    "enhancement_timestamp": datetime.now().isoformat()
                },
                "success": False
            }
    
    @handle_errors(
        category=ErrorCategory.AI_SERVICE,
        level=ErrorLevel.MEDIUM,
        user_message="RAG 기반 코드 생성 중 오류가 발생했습니다."
    )
    async def generate_code_with_rag(
        self,
        user_prompt: str,
        project_id: str = "default",
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인을 통해 컨텍스트가 포함된 코드 생성
        
        Args:
            user_prompt: 사용자 요청
            project_id: 프로젝트 ID
            context_limit: 컨텍스트 개수
            
        Returns:
            생성된 코드와 관련 정보
        """
        try:
            # 1. 향상된 프롬프트 생성
            enhanced_result = await self.generate_enhanced_prompt(
                user_prompt=user_prompt,
                project_id=project_id,
                context_limit=context_limit
            )
            
            if not enhanced_result["success"]:
                return enhanced_result
            
            # 2. LLM 체인 실행 (OpenAI와 DeepSeek 모두 체인 사용)
            if self.llm_chain:
                # LangChain 체인 사용 (OpenAI 또는 DeepSeek)
                response = await self.llm_chain.arun(
                    context=enhanced_result["context"],
                    question=user_prompt
                )
            elif self.llm:
                # 체인 생성에 실패한 경우 직접 호출
                response = await self.llm.arun(
                    context=enhanced_result["context"],
                    question=user_prompt
                )
            else:
                # LLM이 없는 경우 향상된 프롬프트만 반환
                response = f"Enhanced prompt with context (embeddings only):\n\n{enhanced_result['enhanced_prompt']}"
            
            # 3. 결과 구성
            result = {
                "generated_code": response,
                "enhanced_prompt": enhanced_result["enhanced_prompt"],
                "original_prompt": user_prompt,
                "context": enhanced_result["context"],
                "metadata": enhanced_result["metadata"],
                "success": True
            }
            
            logger.info(f"RAG 기반 코드 생성 완료: {len(response)} 문자")
            return result
            
        except Exception as e:
            logger.error(f"RAG 기반 코드 생성 중 오류: {str(e)}")
            return {
                "generated_code": "",
                "enhanced_prompt": user_prompt,
                "original_prompt": user_prompt,
                "context": "",
                "metadata": {
                    "error": str(e),
                    "generation_timestamp": datetime.now().isoformat()
                },
                "success": False
            }
    
    @handle_errors(
        category=ErrorCategory.DATABASE,
        level=ErrorLevel.MEDIUM,
        user_message="문서 인덱싱 중 오류가 발생했습니다."
    )
    async def index_documents_from_directory(
        self,
        directory_path: str,
        project_id: str = "default",
        file_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        디렉토리의 문서들을 인덱싱하여 벡터 DB에 저장
        
        Args:
            directory_path: 인덱싱할 디렉토리 경로
            project_id: 프로젝트 ID
            file_extensions: 처리할 파일 확장자 목록
            
        Returns:
            인덱싱 결과
        """
        try:
            if file_extensions is None:
                file_extensions = ['.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json']
            
            # 디렉토리 로더 생성
            loader = DirectoryLoader(
                directory_path,
                glob="**/*",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            
            # 문서 로드
            documents = loader.load()
            
            # 파일 확장자 필터링
            filtered_docs = []
            for doc in documents:
                file_path = doc.metadata.get('source', '')
                if any(file_path.endswith(ext) for ext in file_extensions):
                    filtered_docs.append(doc)
            
            # 텍스트 분할
            split_docs = self.text_splitter.split_documents(filtered_docs)
            
            # 벡터 DB에 저장
            stored_count = 0
            for doc in split_docs:
                try:
                    # 문서 메타데이터 구성
                    doc_metadata = {
                        'source': doc.metadata.get('source', ''),
                        'type': 'document',
                        'project_id': project_id,
                        'chunk_index': stored_count,
                        'total_chunks': len(split_docs),
                        'indexed_at': datetime.now().isoformat()
                    }
                    
                    # 벡터 서비스에 저장
                    success = await self.vector_service.store_document(
                        content=doc.page_content,
                        metadata=doc_metadata,
                        project_id=project_id
                    )
                    
                    if success:
                        stored_count += 1
                        
                except Exception as e:
                    logger.warning(f"문서 저장 중 오류 (계속 진행): {str(e)}")
                    continue
            
            result = {
                "total_documents": len(documents),
                "filtered_documents": len(filtered_docs),
                "total_chunks": len(split_docs),
                "stored_chunks": stored_count,
                "project_id": project_id,
                "directory_path": directory_path,
                "file_extensions": file_extensions,
                "indexing_timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            logger.info(f"문서 인덱싱 완료: {stored_count}/{len(split_docs)} 청크 저장")
            return result
            
        except Exception as e:
            logger.error(f"문서 인덱싱 중 오류: {str(e)}")
            return {
                "total_documents": 0,
                "filtered_documents": 0,
                "total_chunks": 0,
                "stored_chunks": 0,
                "project_id": project_id,
                "directory_path": directory_path,
                "error": str(e),
                "indexing_timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def search_and_summarize(
        self,
        query: str,
        project_id: str = "default",
        limit: int = 3
    ) -> Dict[str, Any]:
        """
        검색 후 요약 생성
        
        Args:
            query: 검색 쿼리
            project_id: 프로젝트 ID
            limit: 검색 결과 개수
            
        Returns:
            검색 결과와 요약
        """
        try:
            # 리트리버 생성
            retriever = CustomChromaRetriever(
                vector_service=self.vector_service,
                project_id=project_id,
                k=limit
            )
            
            # 관련 문서 검색
            relevant_docs = retriever.get_relevant_documents(query)
            
            # 컨텍스트 구성
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 요약 프롬프트 템플릿
            summary_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""Based on the following context, provide a concise summary that answers the query:

Context:
{context}

Query: {query}

Summary:"""
            )
            
            # 요약 생성
            summary_chain = LLMChain(
                llm=self.llm,
                prompt=summary_template
            )
            
            summary = await summary_chain.arun(
                context=context,
                query=query
            )
            
            return {
                "query": query,
                "summary": summary,
                "relevant_documents": [
                    {
                        "source": doc.metadata.get('source', ''),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "score": doc.metadata.get('score', 0.0)
                    }
                    for doc in relevant_docs
                ],
                "document_count": len(relevant_docs),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"검색 및 요약 중 오류: {str(e)}")
            return {
                "query": query,
                "summary": "",
                "relevant_documents": [],
                "document_count": 0,
                "error": str(e),
                "success": False
            } 