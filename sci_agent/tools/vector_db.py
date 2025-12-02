"""
向量数据库 - 使用Qwen3-Embedding + FAISS进行多模态混合检索
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SearchResult:
    """检索结果"""
    doc_id: str
    source: str
    page: int
    text: str
    score: float
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "score": self.score,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata or {}
        }


class QwenEmbedding:
    """Qwen3-Embedding 向量编码器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        self._dim = 1024
        self._loading = False
        self._loaded = False
    
    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def preload(self):
        """预加载模型（可在启动时调用）"""
        if not self._loaded and not self._loading:
            self._load_model()
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
        
        self._loading = True
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # 设置 HuggingFace 镜像（如果配置了）
            hf_endpoint = os.environ.get("HF_ENDPOINT")
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
                print(f"[Info] 使用 HuggingFace 镜像: {hf_endpoint}")
            
            print(f"[Info] 正在加载 Embedding 模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 获取实际维度
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", padding=True)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                if hasattr(test_output, 'last_hidden_state'):
                    self._dim = test_output.last_hidden_state.shape[-1]
                elif hasattr(test_output, 'pooler_output'):
                    self._dim = test_output.pooler_output.shape[-1]
            
            self._loaded = True
            print(f"[Info] Embedding 模型加载完成，维度: {self._dim}")
        except Exception as e:
            print(f"[Warning] 无法加载Qwen3-Embedding模型: {e}")
            print("[Info] 使用简单的词袋向量作为回退方案")
            self.model = "fallback"
            self._loaded = True
        finally:
            self._loading = False
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组 [N, dim]
        """
        self._load_model()
        
        if self.model == "fallback":
            return self._encode_fallback(texts)
        
        import torch
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 获取句子向量 (mean pooling)
                if hasattr(outputs, 'last_hidden_state'):
                    attention_mask = inputs['attention_mask']
                    hidden_state = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = outputs.pooler_output
                
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """回退方案：简单的词袋向量"""
        import re
        
        # 构建词表
        all_tokens = set()
        for text in texts:
            tokens = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
            all_tokens.update(tokens)
        
        vocab = {w: i for i, w in enumerate(sorted(all_tokens))}
        dim = min(len(vocab), self._dim)
        
        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
            for token in tokens:
                if token in vocab:
                    idx = vocab[token] % self._dim
                    embeddings[i, idx] += 1
            
            # L2归一化
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        
        return embeddings


class VectorDB:
    """
    向量数据库 - 支持多模态混合检索
    
    特性：
    - Qwen3-Embedding 向量编码
    - FAISS 高效检索
    - 支持文本、表格、图像混合检索
    - 支持重排序
    """
    
    def __init__(self, 
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 index_path: Optional[str] = None):
        self.embed_model_name = embed_model
        self.index_path = Path(index_path) if index_path else None
        
        self.encoder = QwenEmbedding(embed_model)
        self.index = None
        self.docs: List[Dict[str, Any]] = []
        self._faiss_available = self._check_faiss()
    
    def _check_faiss(self) -> bool:
        try:
            import faiss
            return True
        except ImportError:
            return False
    
    def index_documents(self, items: List[Dict[str, Any]], incremental: bool = False) -> None:
        """
        索引文档
        
        Args:
            items: 文档块列表，每个包含 text, doc_id, source, page 等字段
            incremental: 是否增量添加（而非重建索引）
        """
        if not items:
            return
        
        # 增量模式：追加到现有文档
        if incremental and self.docs:
            # 过滤已存在的文档（基于 doc_id + page + text 前50字符）
            existing_keys = {
                f"{d.get('doc_id')}_{d.get('page')}_{d.get('text', '')[:50]}"
                for d in self.docs
            }
            new_items = [
                item for item in items
                if f"{item.get('doc_id')}_{item.get('page')}_{item.get('text', '')[:50]}" not in existing_keys
            ]
            if not new_items:
                return
            items = new_items
            self.docs.extend(items)
        else:
            self.docs = items
        
        texts = [item.get("text", "") for item in items]
        
        # 编码向量
        embeddings = self.encoder.encode(texts)
        
        if self._faiss_available:
            import faiss
            
            dim = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            
            if incremental and self.index is not None:
                # 增量添加到现有索引
                self.index.add(embeddings.astype(np.float32))
            else:
                # 创建新索引
                self.index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype(np.float32))
        else:
            # 回退：保存原始向量
            if incremental and self.index is not None:
                self.index = np.vstack([self.index, embeddings])
            else:
                self.index = embeddings
        
        # 保存索引
        if self.index_path:
            self._save_index()
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               filter_type: Optional[str] = None) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_type: 过滤类型 (text/table/image)
            
        Returns:
            检索结果列表
        """
        if not self.docs:
            return []
        
        # 编码查询
        query_vec = self.encoder.encode([query])[0]
        
        if self._faiss_available and self.index is not None:
            import faiss
            
            query_vec = query_vec.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vec)
            
            # 检索更多结果用于过滤
            search_k = top_k * 3 if filter_type else top_k
            scores, indices = self.index.search(query_vec, min(search_k, len(self.docs)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = self.docs[idx]
                
                # 类型过滤
                if filter_type and doc.get("chunk_type") != filter_type:
                    continue
                
                results.append(SearchResult(
                    doc_id=doc.get("doc_id", ""),
                    source=doc.get("source", ""),
                    page=doc.get("page", 0),
                    text=doc.get("text", ""),
                    score=float(score),
                    chunk_type=doc.get("chunk_type", "text"),
                    metadata={k: v for k, v in doc.items() 
                             if k not in ["doc_id", "source", "page", "text", "chunk_type"]}
                ))
                
                if len(results) >= top_k:
                    break
        else:
            # 回退：暴力搜索
            results = self._brute_force_search(query_vec, top_k, filter_type)
        
        return results
    
    def _brute_force_search(self, 
                           query_vec: np.ndarray, 
                           top_k: int,
                           filter_type: Optional[str]) -> List[SearchResult]:
        """暴力搜索回退"""
        if self.index is None:
            return []
        
        # 计算余弦相似度
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_vecs = self.index / (np.linalg.norm(self.index, axis=1, keepdims=True) + 1e-9)
        scores = np.dot(doc_vecs, query_vec)
        
        # 排序
        indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in indices:
            doc = self.docs[idx]
            
            if filter_type and doc.get("chunk_type") != filter_type:
                continue
            
            results.append(SearchResult(
                doc_id=doc.get("doc_id", ""),
                source=doc.get("source", ""),
                page=doc.get("page", 0),
                text=doc.get("text", ""),
                score=float(scores[idx]),
                chunk_type=doc.get("chunk_type", "text"),
                metadata={k: v for k, v in doc.items() 
                         if k not in ["doc_id", "source", "page", "text", "chunk_type"]}
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def hybrid_search(self,
                      query: str,
                      top_k: int = 5,
                      text_weight: float = 0.7,
                      keyword_weight: float = 0.3) -> List[SearchResult]:
        """
        混合检索：向量检索 + 关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            text_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            
        Returns:
            检索结果列表
        """
        # 向量检索
        vector_results = self.search(query, top_k=top_k * 2)
        
        # 关键词检索
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        # 融合结果
        score_map: Dict[str, Tuple[float, Dict]] = {}
        
        for r in vector_results:
            key = f"{r.doc_id}_{r.page}_{r.text[:50]}"
            score_map[key] = (r.score * text_weight, r.to_dict())
        
        for r in keyword_results:
            key = f"{r.doc_id}_{r.page}_{r.text[:50]}"
            if key in score_map:
                old_score, data = score_map[key]
                score_map[key] = (old_score + r.score * keyword_weight, data)
            else:
                score_map[key] = (r.score * keyword_weight, r.to_dict())
        
        # 排序
        sorted_results = sorted(score_map.items(), key=lambda x: x[1][0], reverse=True)
        
        results = []
        for _, (score, data) in sorted_results[:top_k]:
            results.append(SearchResult(
                doc_id=data["doc_id"],
                source=data["source"],
                page=data["page"],
                text=data["text"],
                score=score,
                chunk_type=data.get("chunk_type", "text"),
                metadata=data.get("metadata", {})
            ))
        
        return results
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """关键词检索"""
        import re
        
        query_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))
        
        scored = []
        for doc in self.docs:
            text = doc.get("text", "")
            doc_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', text.lower()))
            
            # Jaccard相似度
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            score = intersection / union if union > 0 else 0
            
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scored[:top_k]:
            results.append(SearchResult(
                doc_id=doc.get("doc_id", ""),
                source=doc.get("source", ""),
                page=doc.get("page", 0),
                text=doc.get("text", ""),
                score=score,
                chunk_type=doc.get("chunk_type", "text")
            ))
        
        return results
    
    def _save_index(self) -> None:
        """保存索引到磁盘"""
        if not self.index_path:
            return
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存文档元数据
        meta_path = self.index_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False)
        
        # 保存索引
        if self._faiss_available and self.index is not None:
            import faiss
            faiss.write_index(self.index, str(self.index_path))
            print(f"[Info] 已保存 FAISS 索引: {self.index_path}")
        elif self.index is not None:
            npy_path = str(self.index_path) + ".npy"
            np.save(npy_path, self.index)
            print(f"[Info] 已保存 NumPy 索引: {npy_path}")
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        if not self.index_path:
            return False
        
        meta_path = self.index_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"[Warning] 索引元数据文件不存在: {meta_path}")
            return False
        
        # 加载文档元数据
        with open(meta_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        
        # 加载索引 - 检查多种可能的文件格式
        faiss_path = self.index_path
        npy_path = Path(str(self.index_path) + ".npy")
        
        if self._faiss_available and faiss_path.exists():
            import faiss
            self.index = faiss.read_index(str(faiss_path))
            print(f"[Info] 已加载 FAISS 索引，共 {len(self.docs)} 个文档块")
        elif npy_path.exists():
            self.index = np.load(str(npy_path))
            print(f"[Info] 已加载 NumPy 索引，共 {len(self.docs)} 个文档块")
        else:
            print(f"[Warning] 索引文件不存在: {faiss_path} 或 {npy_path}")
            return False
        
        return True
