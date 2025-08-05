"""
고급 분석 서비스 - scikit-learn을 활용한 프롬프트 분석
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# scikit-learn 임포트
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedAnalyticsService:
    """고급 분석 서비스"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.kmeans_model = None
        self.pca_model = None
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 설치되지 않았습니다. 고급 분석 기능이 제한됩니다.")
    
    async def calculate_advanced_similarity(
        self, 
        embeddings1: List[float], 
        embeddings2: List[float]
    ) -> float:
        """고급 유사도 계산 (코사인 유사도)"""
        if not SKLEARN_AVAILABLE:
            # 기본 내적 기반 유사도
            return self._dot_product_similarity(embeddings1, embeddings2)
        
        try:
            # scikit-learn 코사인 유사도
            emb1 = np.array(embeddings1).reshape(1, -1)
            emb2 = np.array(embeddings2).reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"고급 유사도 계산 실패: {e}")
            return self._dot_product_similarity(embeddings1, embeddings2)
    
    async def cluster_prompts(
        self, 
        prompt_embeddings: List[List[float]], 
        prompt_texts: List[str],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """프롬프트 클러스터링"""
        if not SKLEARN_AVAILABLE or len(prompt_embeddings) < n_clusters:
            return {
                "clusters": [],
                "labels": [],
                "cluster_centers": [],
                "silhouette_score": 0.0
            }
        
        try:
            embeddings_array = np.array(prompt_embeddings)
            
            # KMeans 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            
            # 클러스터별 대표 프롬프트 추출
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                cluster_prompts = [prompt_texts[idx] for idx in cluster_indices]
                
                # 클러스터 중심에서 가장 가까운 프롬프트를 대표로 선택
                cluster_embeddings = embeddings_array[cluster_indices]
                center = kmeans.cluster_centers_[i]
                
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                
                clusters.append({
                    "cluster_id": int(i),
                    "size": len(cluster_prompts),
                    "representative_prompt": prompt_texts[representative_idx],
                    "prompts": cluster_prompts[:5],  # 상위 5개만
                    "center": center.tolist()
                })
            
            # 실루엣 점수 계산
            try:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(embeddings_array, labels)
            except:
                silhouette_avg = 0.0
            
            return {
                "clusters": clusters,
                "labels": labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "silhouette_score": float(silhouette_avg)
            }
            
        except Exception as e:
            logger.error(f"프롬프트 클러스터링 실패: {e}")
            return {
                "clusters": [],
                "labels": [],
                "cluster_centers": [],
                "silhouette_score": 0.0
            }
    
    async def extract_text_features(
        self, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """TF-IDF를 사용한 텍스트 특성 추출"""
        if not SKLEARN_AVAILABLE:
            return {"features": [], "vocabulary": [], "feature_names": []}
        
        try:
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 중요한 특성 추출
            feature_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_features_idx = np.argsort(feature_scores)[::-1][:20]
            
            top_features = [
                {
                    "term": feature_names[idx],
                    "score": float(feature_scores[idx])
                }
                for idx in top_features_idx
            ]
            
            return {
                "tfidf_matrix": tfidf_matrix.toarray().tolist(),
                "feature_names": feature_names.tolist(),
                "top_features": top_features,
                "vocabulary_size": len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"텍스트 특성 추출 실패: {e}")
            return {"features": [], "vocabulary": [], "feature_names": []}
    
    async def reduce_dimensions(
        self, 
        embeddings: List[List[float]], 
        method: str = "pca",
        n_components: int = 2
    ) -> Dict[str, Any]:
        """차원 축소 (시각화용)"""
        if not SKLEARN_AVAILABLE:
            return {"reduced_embeddings": [], "explained_variance": []}
        
        try:
            embeddings_array = np.array(embeddings)
            
            if method.lower() == "pca":
                reducer = PCA(n_components=n_components)
                reduced = reducer.fit_transform(embeddings_array)
                explained_variance = reducer.explained_variance_ratio_.tolist()
            elif method.lower() == "tsne":
                reducer = TSNE(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings_array)
                explained_variance = []
            else:
                raise ValueError(f"지원하지 않는 차원 축소 방법: {method}")
            
            return {
                "reduced_embeddings": reduced.tolist(),
                "explained_variance": explained_variance,
                "method": method,
                "original_dimensions": embeddings_array.shape[1],
                "reduced_dimensions": n_components
            }
            
        except Exception as e:
            logger.error(f"차원 축소 실패: {e}")
            return {"reduced_embeddings": [], "explained_variance": []}
    
    async def analyze_prompt_trends(
        self, 
        prompt_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """프롬프트 트렌드 분석"""
        try:
            # 시간별 분석
            time_analysis = self._analyze_temporal_patterns(prompt_data)
            
            # 길이 분석
            length_analysis = self._analyze_prompt_lengths(prompt_data)
            
            # 복잡도 분석
            complexity_analysis = self._analyze_prompt_complexity(prompt_data)
            
            return {
                "temporal_patterns": time_analysis,
                "length_distribution": length_analysis,
                "complexity_metrics": complexity_analysis,
                "total_prompts": len(prompt_data)
            }
            
        except Exception as e:
            logger.error(f"프롬프트 트렌드 분석 실패: {e}")
            return {}
    
    def _dot_product_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """기본 내적 기반 유사도"""
        try:
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _analyze_temporal_patterns(self, prompt_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시간 패턴 분석"""
        if not prompt_data:
            return {}
        
        # 시간대별 분포 등 분석
        time_counts = {}
        for prompt in prompt_data:
            created_at = prompt.get('created_at')
            if created_at:
                try:
                    if isinstance(created_at, str):
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        dt = created_at
                    hour = dt.hour
                    time_counts[hour] = time_counts.get(hour, 0) + 1
                except:
                    continue
        
        return {
            "hourly_distribution": time_counts,
            "peak_hour": max(time_counts.items(), key=lambda x: x[1])[0] if time_counts else 0,
            "total_hours": len(time_counts)
        }
    
    def _analyze_prompt_lengths(self, prompt_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """프롬프트 길이 분석"""
        lengths = []
        for prompt in prompt_data:
            content = prompt.get('content', '')
            lengths.append(len(content))
        
        if not lengths:
            return {}
        
        return {
            "average_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "median_length": sorted(lengths)[len(lengths) // 2]
        }
    
    def _analyze_prompt_complexity(self, prompt_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """프롬프트 복잡도 분석"""
        complexities = []
        
        for prompt in prompt_data:
            content = prompt.get('content', '')
            
            # 간단한 복잡도 지표들
            word_count = len(content.split())
            sentence_count = content.count('.') + content.count('!') + content.count('?')
            
            complexity = {
                "word_count": word_count,
                "sentence_count": max(1, sentence_count),
                "avg_words_per_sentence": word_count / max(1, sentence_count)
            }
            complexities.append(complexity)
        
        if not complexities:
            return {}
        
        avg_words = sum(c["word_count"] for c in complexities) / len(complexities)
        avg_sentences = sum(c["sentence_count"] for c in complexities) / len(complexities)
        
        return {
            "average_word_count": avg_words,
            "average_sentence_count": avg_sentences,
            "average_words_per_sentence": avg_words / max(1, avg_sentences)
        } 