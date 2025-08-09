import uuid
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from models.prompt_models import PromptHistory, ProjectContext
from config import settings
import asyncio
import re
from datetime import datetime, timedelta
import time

# 선택적 임포트
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    CHROMADB_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        OpenAIEmbeddings = None

# DeepSeek R1 임베딩을 위한 HTTP 클라이언트
import aiohttp
import json

# LangChain Runnable 인터페이스
try:
    from langchain.schema.runnable import Runnable
    from langchain.schema.output_parser import BaseOutputParser
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    LANGCHAIN_RUNNABLE_AVAILABLE = True
except ImportError:
    Runnable = object  # fallback
    BaseOutputParser = object
    AsyncCallbackManagerForLLMRun = None
    CallbackManagerForLLMRun = None
    LANGCHAIN_RUNNABLE_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

# 고급 검색을 위한 추가 임포트
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeepSeekEmbeddings:
    """DeepSeek R1 임베딩 클래스"""
    
    def __init__(self, api_base: str, model_name: str):
        self.api_base = api_base
        self.model_name = model_name
        self.session = None
        # 연결 풀 설정 개선
        self.connector = None
        self.max_batch_size = 50  # 배치 크기 제한
        
    async def _get_session(self):
        """HTTP 세션 가져오기 (연결 풀링 최적화)"""
        if self.session is None:
            # 연결 풀 설정 최적화
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=100,  # 전체 연결 풀 크기
                    limit_per_host=50,  # 호스트당 연결 수
                    ttl_dns_cache=300,  # DNS 캐시 TTL
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
            
            timeout = aiohttp.ClientTimeout(
                total=120,  # 전체 타임아웃 증가
                connect=10,
                sock_read=30
            )
            
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout
            )
        return self.session
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 생성 (배치 최적화)"""
        if not texts:
            return []
            
        try:
            session = await self._get_session()
            embeddings = []
            
            # 배치 크기로 나누어 처리
            for i in range(0, len(texts), self.max_batch_size):
                batch_texts = texts[i:i + self.max_batch_size]
                
                # 배치 내에서 병렬 처리
                batch_tasks = []
                for text in batch_texts:
                    task = self._embed_single_text(session, text)
                    batch_tasks.append(task)
                
                # 배치 병렬 실행
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"DeepSeek 임베딩 생성 실패: {result}")
                        embeddings.append([0.0] * 768)
                    else:
                        embeddings.append(result)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"DeepSeek 임베딩 생성 중 오류: {e}")
            return [[0.0] * 768] * len(texts)
    
    async def _embed_single_text(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        try:
            async with session.post(
                f"{self.api_base}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text[:8192]  # 텍스트 길이 제한
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("embedding", [0.0] * 768)
                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek 임베딩 API 오류 {response.status}: {error_text}")
                    return [0.0] * 768
                    
        except aiohttp.ClientError as e:
            logger.error(f"네트워크 오류 발생: {e}")
            return [0.0] * 768
        except asyncio.TimeoutError:
            logger.error("임베딩 생성 타임아웃")
            return [0.0] * 768
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return [0.0] * 768
    
    async def aembed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 생성"""
        embeddings = await self.aembed_documents([text])
        return embeddings[0] if embeddings else [0.0] * 768
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """동기 버전 (호환성용)"""
        return asyncio.run(self.aembed_documents(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """동기 버전 (호환성용)"""
        return asyncio.run(self.aembed_query(text))
    
    async def close(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
            self.session = None
        if self.connector:
            await self.connector.close()
            self.connector = None
    
    def __del__(self):
        """소멸자"""
        try:
            if self.session or self.connector:
                try:
                    loop = asyncio.get_running_loop()
                    if not loop.is_closed():
                        loop.create_task(self.close())
                except RuntimeError:
                    # 이벤트 루프가 없거나 닫혀있음 - 정상적인 상황
                    pass
        except Exception:
            pass


class DeepSeekLLM(Runnable):
    """DeepSeek R1 LLM 클래스 (LangChain Runnable 호환)"""
    
    def __init__(self, api_base: str, model_name: str, temperature: float = 0.1, max_tokens: int = 4000):
        super().__init__()
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = None
        
    async def _get_session(self):
        """HTTP 세션 가져오기"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def agenerate(self, prompts: List[str]) -> List[str]:
        """텍스트 생성 (비동기)"""
        try:
            session = await self._get_session()
            results = []
            
            for prompt in prompts:
                async with session.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results.append(result.get("response", ""))
                    else:
                        logger.error(f"DeepSeek LLM 생성 실패: {response.status}")
                        results.append("Error: Failed to generate response")
            
            return results
        except Exception as e:
            logger.error(f"DeepSeek LLM 생성 중 오류: {e}")
            return [f"Error: {str(e)}"] * len(prompts)
    
    async def arun(self, **kwargs) -> str:
        """단일 프롬프트 실행 (LangChain 호환)"""
        # 프롬프트 템플릿에서 전달된 변수들을 하나의 프롬프트로 조합
        if 'context' in kwargs and 'question' in kwargs:
            prompt = f"""# Context from my project
{kwargs['context']}

---
Based on the context above, please write code to satisfy the following requirement.

Requirement: {kwargs['question']}

Please provide:
1. A complete, working code solution
2. Brief explanation of the implementation
3. Any necessary imports or dependencies
4. Usage examples if applicable

Focus on using the project context to maintain consistency with existing code patterns, naming conventions, and architecture."""
        else:
            # 다른 형태의 kwargs가 올 경우 문자열로 변환
            prompt = str(kwargs)
        
        results = await self.agenerate([prompt])
        return results[0] if results else "Error: No response generated"
    
    def generate(self, prompts: List[str]) -> List[str]:
        """동기 버전 (호환성용)"""
        return asyncio.run(self.agenerate(prompts))
    
    def run(self, **kwargs) -> str:
        """동기 버전 (호환성용)"""
        return asyncio.run(self.arun(**kwargs))
    
    async def ainvoke(self, input_data, config=None, **kwargs) -> str:
        """LangChain Runnable 인터페이스 구현 (비동기)"""
        if isinstance(input_data, str):
            # 단순 문자열 입력
            return await self.arun(prompt=input_data)
        elif isinstance(input_data, dict):
            # 딕셔너리 입력 (프롬프트 템플릿에서 오는 경우)
            return await self.arun(**input_data)
        else:
            # 기타 형태
            return await self.arun(prompt=str(input_data))
    
    def invoke(self, input_data, config=None, **kwargs) -> str:
        """LangChain Runnable 인터페이스 구현 (동기)"""
        return asyncio.run(self.ainvoke(input_data, config, **kwargs))
    
    def stream(self, input_data, config=None, **kwargs):
        """스트리밍 지원 (기본 구현)"""
        result = self.invoke(input_data, config, **kwargs)
        yield result
    
    async def astream(self, input_data, config=None, **kwargs):
        """비동기 스트리밍 지원 (기본 구현)"""
        result = await self.ainvoke(input_data, config, **kwargs)
        yield result
    
    def batch(self, inputs, config=None, **kwargs):
        """배치 처리 지원"""
        return [self.invoke(input_data, config, **kwargs) for input_data in inputs]
    
    async def abatch(self, inputs, config=None, **kwargs):
        """비동기 배치 처리 지원"""
        tasks = [self.ainvoke(input_data, config, **kwargs) for input_data in inputs]
        return await asyncio.gather(*tasks)
    
    def __call__(self, prompt: str) -> str:
        """직접 호출 가능하도록"""
        return self.invoke(prompt)
    
    async def close(self):
        """세션 정리"""
        if self.session:
            await self.session.close()
            self.session = None

class VectorService:
    """벡터 데이터베이스 서비스"""
    
    def __init__(self):
        # 임베딩 모델 초기화 (config에 따라 선택)
        self.embeddings = None
        self.embedding_model_type = os.getenv("EMBEDDING_MODEL_TYPE", settings.embedding_model_type)
        
        if self.embedding_model_type == "deepseek":
            try:
                deepseek_api_base = os.getenv("DEEPSEEK_API_BASE", settings.deepseek_api_base)
                deepseek_embedding_model = os.getenv("DEEPSEEK_EMBEDDING_MODEL", settings.deepseek_embedding_model)
                
                self.embeddings = DeepSeekEmbeddings(
                    api_base=deepseek_api_base,
                    model_name=deepseek_embedding_model
                )
                logger.info(f"DeepSeek 임베딩 초기화 완료: {deepseek_api_base} (모델: {deepseek_embedding_model})")
            except Exception as e:
                logger.warning(f"DeepSeek 임베딩 초기화 실패: {e}")
        
        elif self.embedding_model_type == "openai":
            if settings.openai_api_key and OpenAIEmbeddings:
                try:
                    self.embeddings = OpenAIEmbeddings(
                        openai_api_key=settings.openai_api_key
                    )
                    logger.info("OpenAI 임베딩 초기화 완료")
                except Exception as e:
                    logger.warning(f"OpenAI 임베딩 초기화 실패: {e}")
        
        else:
            logger.warning(f"알 수 없는 임베딩 모델 타입: {self.embedding_model_type}")
        
        # ChromaDB는 선택적으로 사용
        self.chroma_client = None
        self.prompt_collection = None
        self.context_collection = None
        
        # 하이브리드 검색을 위한 TF-IDF 인덱스 및 벡터라이저
        self.tfidf_vectorizer = None
        self._tfidf_index_cache = {}
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=8000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # 검색 성능 최적화를 위한 캐시
        self.search_cache = {}
        self.cache_ttl = 300  # 5분
        
        # ChromaDB 초기화 시도
        if CHROMADB_AVAILABLE:
            try:
                # config.py와 환경변수로 ChromaDB 서버 설정 확인
                chroma_host = os.getenv("CHROMA_HOST", settings.chroma_host)
                chroma_port = os.getenv("CHROMA_PORT", str(settings.chroma_port))
                
                # Docker 환경에서는 서버 모드, 로컬에서는 파일 모드
                if chroma_host and chroma_host != "localhost":
                    # 서버 모드: Docker Compose의 ChromaDB 서비스에 연결 (v2 API 호환)
                    logger.info(f"ChromaDB v2 서버에 연결 중: {chroma_host}:{chroma_port}")
                    self.chroma_client = chromadb.HttpClient(
                        host=chroma_host,
                        port=int(chroma_port),
                        ssl=False,  # Docker 내부 통신은 HTTP 사용
                        headers=None,  # v2 API용 헤더
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            chroma_api_impl="rest",  # REST API 명시
                            chroma_server_host=chroma_host,
                            chroma_server_http_port=int(chroma_port)
                        )
                    )
                else:
                    # 로컬 모드: 파일 기반 ChromaDB
                    logger.info(f"로컬 ChromaDB 사용: {settings.chroma_db_path}")
                    self.chroma_client = chromadb.PersistentClient(
                        path=settings.chroma_db_path,
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                            is_persistent=True
                        )
                    )
                
                # 컬렉션 초기화
                self.prompt_collection = self._get_or_create_collection(settings.chroma_collection_name)
                self.context_collection = self._get_or_create_collection("project_context")
                logger.info(f"ChromaDB 클라이언트 초기화 완료 (컬렉션: {settings.chroma_collection_name})")
                
            except Exception as e:
                logger.warning(f"ChromaDB 초기화 실패: {e}")
        else:
            logger.warning("ChromaDB가 설치되지 않았습니다. 벡터 검색 기능이 비활성화됩니다.")
        
        # Text Splitter 초기화
        self.text_splitter = None
        if RecursiveCharacterTextSplitter:
            try:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
            except Exception as e:
                logger.warning(f"Text splitter 초기화 실패: {e}")
        else:
            logger.warning("Text splitter를 사용할 수 없습니다.")
    
    def _get_or_create_collection(self, name: str):
        """컬렉션 가져오기 또는 생성"""
        try:
            return self.chroma_client.get_collection(name)
        except Exception:
            return self.chroma_client.create_collection(name)
    
    def _get_cache_key(self, query: str, project_id: str, limit: int, search_type: str = "default") -> str:
        """캐시 키 생성"""
        import hashlib
        key_string = f"{query}:{project_id}:{limit}:{search_type}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """캐시 유효성 확인"""
        return (datetime.now().timestamp() - timestamp) < self.cache_ttl
    
    async def store_prompt_history(self, prompt_history: PromptHistory) -> bool:
        """프롬프트 히스토리 저장"""
        if not self.prompt_collection:
            logger.warning("벡터 데이터베이스가 초기화되지 않았습니다.")
            return False
            
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # 임베딩 생성
                embedding = await self._generate_embedding(prompt_history.content)
                
                # 메타데이터 강화
                enhanced_metadata = {
                    "id": prompt_history.id,
                    "project_id": prompt_history.project_id,
                    "prompt_type": prompt_history.prompt_type.value,
                    "created_at": prompt_history.created_at.isoformat(),
                    "content_length": len(prompt_history.content),
                    "word_count": len(prompt_history.content.split()),
                    "has_code": self._contains_code(prompt_history.content),
                    "complexity_score": self._calculate_text_complexity(prompt_history.content),
                    **prompt_history.metadata
                }
                
                # ChromaDB에 저장
                self.prompt_collection.add(
                    documents=[prompt_history.content],
                    embeddings=[embedding],
                    metadatas=[enhanced_metadata],
                    ids=[prompt_history.id]
                )
                
                # 캐시 무효화
                self.search_cache.clear()
                
                logger.info(f"프롬프트 히스토리 저장 완료: {prompt_history.id}")
                return True
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"프롬프트 히스토리 저장 최종 실패: {e}")
                    return False
                else:
                    logger.warning(f"프롬프트 히스토리 저장 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
    
    async def store_project_context(self, project_context: ProjectContext) -> bool:
        """프로젝트 컨텍스트 저장"""
        try:
            # 프로젝트 설명을 기반으로 임베딩 생성
            context_text = f"{project_context.project_name} {project_context.description or ''} {' '.join(project_context.tech_stack)}"
            embedding = await self._generate_embedding(context_text)
            
            self.context_collection.add(
                documents=[context_text],
                embeddings=[embedding],
                metadatas=[{
                    "project_id": project_context.project_id,
                    "project_name": project_context.project_name,
                    "description": project_context.description or "",
                    "tech_stack": ",".join(project_context.tech_stack),
                    "file_patterns": ",".join(project_context.file_patterns),
                    "created_at": project_context.created_at.isoformat()
                }],
                ids=[project_context.project_id]
            )
            
            logger.info(f"프로젝트 컨텍스트 저장 완료: {project_context.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 컨텍스트 저장 실패: {e}")
            return False
    
    async def search_similar_prompts(
        self, 
        query: str, 
        project_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """유사한 프롬프트 검색 (향상된 하이브리드 검색)"""
        try:
            # 캐시 확인
            cache_key = self._get_cache_key(query, project_id, limit, "hybrid")
            if cache_key in self.search_cache:
                cached_result, timestamp = self.search_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.info("캐시된 검색 결과 사용")
                    return cached_result
            
            # 하이브리드 검색 수행 (병렬)
            semantic_task = asyncio.create_task(self._semantic_search(query, project_id, limit * 2))
            keyword_task = asyncio.create_task(self._keyword_search(query, project_id, limit * 2))
            semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
            
            # 결과 병합 및 재랭킹
            combined_results = self._combine_and_rerank(
                semantic_results, 
                keyword_results, 
                query, 
                limit
            )
            
            # 캐시 저장
            self.search_cache[cache_key] = (combined_results, datetime.now().timestamp())
            
            return combined_results
            
        except Exception as e:
            logger.error(f"유사한 프롬프트 검색 실패: {e}")
            return []
    
    async def _semantic_search(
        self, 
        query: str, 
        project_id: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """의미적 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self._generate_embedding(query)
            
            # 유사 프롬프트 검색
            results = self.prompt_collection.query(
                query_embeddings=[query_embedding],
                where={"project_id": project_id},
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_prompts = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # 거리를 유사도로 변환 (코사인 거리 -> 유사도)
                    similarity = 1 - distance
                    
                    similar_prompts.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "search_type": "semantic"
                    })
            
            return similar_prompts
            
        except Exception as e:
            logger.error(f"의미적 검색 실패: {e}")
            return []
    
    async def _keyword_search(
        self, 
        query: str, 
        project_id: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """키워드 기반 검색"""
        try:
            if not SKLEARN_AVAILABLE:
                return []
            
            # 프로젝트의 모든 프롬프트 가져오기
            all_results = self.prompt_collection.get(
                where={"project_id": project_id},
                include=["documents", "metadatas"]
            )
            
            if not all_results["documents"]:
                return []
            
            documents = all_results["documents"]
            metadatas = all_results["metadatas"]
            
            # TF-IDF 벡터화 (프로젝트별 캐시)
            index_entry = self._tfidf_index_cache.get(project_id)
            now_ts = datetime.now().timestamp()
            index_ttl = getattr(settings, 'tfidf_index_ttl_seconds', 300)
            if not index_entry or now_ts - index_entry.get('built_at', 0) > index_ttl or len(index_entry.get('documents', [])) != len(documents):
                tfidf_matrix_all = self.tfidf_vectorizer.fit_transform(documents)
                self._tfidf_index_cache[project_id] = {
                    'built_at': now_ts,
                    'tfidf_matrix': tfidf_matrix_all,
                    'documents': documents
                }
            else:
                tfidf_matrix_all = index_entry['tfidf_matrix']

            # 쿼리 벡터는 기존 vocabulary에 맞춰 transform
            query_vector = self.tfidf_vectorizer.transform([query])
            doc_vectors = tfidf_matrix_all
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 결과 정렬 및 제한
            ranked_indices = np.argsort(similarities)[::-1][:limit]
            
            keyword_results = []
            for idx in ranked_indices:
                if similarities[idx] > 0.1:  # 최소 유사도 임계값
                    keyword_results.append({
                        "content": documents[idx],
                        "metadata": metadatas[idx],
                        "similarity": float(similarities[idx]),
                        "search_type": "keyword"
                    })
            
            return keyword_results
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {e}")
            return []
    
    def _combine_and_rerank(
        self, 
        semantic_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]], 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """의미적 검색과 키워드 검색 결과를 병합하고 재랭킹"""
        
        # 결과 병합 (ID 기반으로 중복 제거)
        combined_results = {}
        
        # 가중치 설정
        w_sem = getattr(settings, 'hybrid_semantic_weight', 0.7)
        w_kw = getattr(settings, 'hybrid_keyword_weight', 0.3)
        w_rec = getattr(settings, 'recency_weight', 0.1)
        w_cplx = getattr(settings, 'complexity_weight', 0.1)

        # 의미적 검색 결과 추가
        for result in semantic_results:
            result_id = result["metadata"].get("id")
            if result_id:
                combined_results[result_id] = {
                    **result,
                    "semantic_score": result["similarity"],
                    "keyword_score": 0.0,
                    "combined_score": result["similarity"] * w_sem
                }
        
        # 키워드 검색 결과 추가
        for result in keyword_results:
            result_id = result["metadata"].get("id")
            if result_id:
                if result_id in combined_results:
                    # 기존 결과 업데이트
                    combined_results[result_id]["keyword_score"] = result["similarity"]
                    combined_results[result_id]["combined_score"] = (
                        combined_results[result_id]["semantic_score"] * w_sem +
                        result["similarity"] * w_kw
                    )
                else:
                    # 새로운 결과 추가
                    combined_results[result_id] = {
                        **result,
                        "semantic_score": 0.0,
                        "keyword_score": result["similarity"],
                        "combined_score": result["similarity"] * w_kw
                    }
        
        # 추가 랭킹 요소 적용
        for result_id, result in combined_results.items():
            # 시간 기반 점수 (최근 것일수록 높은 점수)
            time_score = self._calculate_time_score(result["metadata"])
            
            # 복잡도 기반 점수
            complexity_score = result["metadata"].get("complexity_score", 0.5)
            
            # 최종 점수 계산
            # 합은 1.0을 넘지 않도록 구성
            result["final_score"] = (
                result["combined_score"] * (1.0 - (w_rec + w_cplx)) +
                time_score * w_rec +
                complexity_score * w_cplx
            )
        
        # 최종 점수로 정렬
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        # 상위 결과 반환
        return sorted_results[:limit]
    
    def _calculate_time_score(self, metadata: Dict[str, Any]) -> float:
        """시간 기반 점수 계산"""
        try:
            created_at = metadata.get("created_at")
            if not created_at:
                return 0.5
            
            created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now()
            
            # 30일 이내의 프롬프트에 대해 시간 점수 계산
            days_diff = (now - created_time).days
            if days_diff <= 30:
                return 1.0 - (days_diff / 30.0)
            else:
                return 0.0
                
        except Exception:
            return 0.5
    
    def _contains_code(self, text: str) -> bool:
        """텍스트에 코드가 포함되어 있는지 확인"""
        code_patterns = [
            r'```',  # 코드 블록
            r'def\s+\w+\s*\(',  # Python 함수
            r'function\s+\w+\s*\(',  # JavaScript 함수
            r'class\s+\w+\s*[\(\{]',  # 클래스 정의
            r'import\s+\w+',  # import 문
            r'from\s+\w+\s+import',  # from import 문
            r'#include\s*<',  # C/C++ include
            r'public\s+static\s+void\s+main'  # Java main
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_patterns)
    
    def _calculate_text_complexity(self, text: str) -> float:
        """텍스트 복잡도 계산"""
        try:
            words = text.split()
            sentences = len([s for s in text.split('.') if s.strip()])
            
            # 기본 복잡도 지표
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            avg_sentence_length = len(words) / sentences if sentences > 0 else 0
            
            # 기술 용어 밀도
            tech_terms = ['function', 'class', 'method', 'variable', 'API', 'database', 'server', 'client']
            tech_density = sum(1 for term in tech_terms if term.lower() in text.lower()) / len(words) if words else 0
            
            # 정규화된 복잡도 점수 (0-1)
            complexity = min(
                (avg_word_length / 10) * 0.3 +
                (avg_sentence_length / 20) * 0.4 +
                tech_density * 0.3,
                1.0
            )
            
            return complexity
            
        except Exception:
            return 0.5
    
    async def get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """프로젝트 컨텍스트 조회"""
        try:
            results = self.context_collection.get(
                ids=[project_id],
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                return {
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"프로젝트 컨텍스트 조회 실패: {e}")
            return None
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        try:
            if self.embeddings:
                # OpenAI 임베딩 사용
                embedding = await self.embeddings.aembed_query(text)
                return embedding
            else:
                # 간단한 더미 임베딩 (실제 환경에서는 다른 임베딩 모델 사용)
                logger.warning("OpenAI 임베딩을 사용할 수 없어 더미 임베딩을 생성합니다.")
                import hashlib
                import struct
                
                # 텍스트 해시를 기반으로 간단한 벡터 생성
                hash_bytes = hashlib.md5(text.encode()).digest()
                vector = []
                for i in range(0, len(hash_bytes), 4):
                    chunk = hash_bytes[i:i+4]
                    if len(chunk) == 4:
                        vector.append(struct.unpack('f', chunk)[0])
                
                # 벡터 크기를 768으로 맞춤 (DeepSeek 임베딩 크기)
                while len(vector) < 768:
                    vector.extend(vector[:min(len(vector), 768 - len(vector))])
                
                return vector[:768]
                
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # 기본 더미 벡터 반환
            return [0.0] * 768
    
    async def delete_project_data(self, project_id: str) -> bool:
        """프로젝트 데이터 삭제"""
        try:
            # 프롬프트 히스토리 삭제
            self.prompt_collection.delete(
                where={"project_id": project_id}
            )
            
            # 프로젝트 컨텍스트 삭제
            self.context_collection.delete(
                ids=[project_id]
            )
            
            # 캐시 무효화
            self.search_cache.clear()
            
            logger.info(f"프로젝트 데이터 삭제 완료: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 데이터 삭제 실패: {e}")
            return False
    
    async def get_search_statistics(self, project_id: str) -> Dict[str, Any]:
        """검색 통계 정보 조회"""
        try:
            # 프롬프트 통계
            prompt_results = self.prompt_collection.get(
                where={"project_id": project_id},
                include=["metadatas"]
            )
            
            if not prompt_results["metadatas"]:
                return {
                    "total_prompts": 0,
                    "cache_size": len(self.search_cache),
                    "avg_complexity": 0.0
                }
            
            metadatas = prompt_results["metadatas"]
            
            # 통계 계산
            total_prompts = len(metadatas)
            complexity_scores = [m.get("complexity_score", 0.5) for m in metadatas]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            # 타입별 분포
            type_counts = {}
            for metadata in metadatas:
                prompt_type = metadata.get("prompt_type", "unknown")
                type_counts[prompt_type] = type_counts.get(prompt_type, 0) + 1
            
            return {
                "total_prompts": total_prompts,
                "cache_size": len(self.search_cache),
                "avg_complexity": avg_complexity,
                "type_distribution": type_counts,
                "search_cache_hits": getattr(self, '_cache_hits', 0),
                "search_cache_misses": getattr(self, '_cache_misses', 0)
            }
            
        except Exception as e:
            logger.error(f"검색 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def optimize_search_cache(self):
        """검색 캐시 최적화"""
        try:
            current_time = datetime.now().timestamp()
            
            # 만료된 캐시 항목 제거
            expired_keys = [
                key for key, (_, timestamp) in self.search_cache.items()
                if not self._is_cache_valid(timestamp)
            ]
            
            for key in expired_keys:
                del self.search_cache[key]
            
            logger.info(f"검색 캐시 최적화 완료: {len(expired_keys)}개 항목 삭제")
            
        except Exception as e:
            logger.error(f"검색 캐시 최적화 실패: {e}") 

    async def store_chunks(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> bool:
        """청크들을 벡터 DB에 저장 (성능 최적화)"""
        try:
            if not chunks or not metadata_list:
                return False
            
            if len(chunks) != len(metadata_list):
                logger.error("청크와 메타데이터 개수가 일치하지 않습니다")
                return False
            
            await self._ensure_chroma_initialized()
            
            if not self.prompt_collection:
                logger.error("ChromaDB 컬렉션이 초기화되지 않았습니다")
                return False
            
            # 대용량 배치를 작은 청크로 분할하여 처리 (ChromaDB 제한 고려)
            batch_size = 500  # 한 번에 저장할 청크 수 증가
            total_stored = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_metadata = metadata_list[i:i + batch_size]
                
                # 임베딩 생성 (배치 처리)
                logger.info(f"배치 {i//batch_size + 1} 임베딩 생성 중... ({len(batch_chunks)}개)")
                embeddings = await self._generate_batch_embeddings(batch_chunks)
                
                if not embeddings or len(embeddings) != len(batch_chunks):
                    logger.error(f"임베딩 생성 실패: {len(embeddings)} != {len(batch_chunks)}")
                    continue
                
                # ChromaDB에 배치 저장
                try:
                    ids = [f"chunk_{int(time.time() * 1000000)}_{j}" for j in range(len(batch_chunks))]
                    
                    self.prompt_collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadata,
                        embeddings=embeddings,
                        ids=ids
                    )
                    
                    total_stored += len(batch_chunks)
                    logger.info(f"배치 저장 완료: {len(batch_chunks)}개 청크")
                    
                    # 메모리 압박 방지
                    if i % (batch_size * 2) == 0 and i > 0:
                        await asyncio.sleep(0.05)  # 50ms 대기
                        
                except Exception as e:
                    logger.error(f"ChromaDB 배치 저장 실패: {e}")
                    continue
            
            logger.info(f"총 {total_stored}개 청크 저장 완료")
            return total_stored > 0
            
        except Exception as e:
            logger.error(f"청크 저장 중 오류: {e}")
            return False
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성 (최적화)"""
        try:
            if self.embeddings:
                return await self.embeddings.aembed_documents(texts)
            else:
                # 폴백: 병렬 개별 임베딩
                semaphore = asyncio.Semaphore(50)  # 높은 동시성
                
                async def embed_single(text):
                    async with semaphore:
                        return await self._generate_embedding(text)
                
                return await asyncio.gather(*[embed_single(text) for text in texts])
                
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            return [[0.0] * 768] * len(texts) 