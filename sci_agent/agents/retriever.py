"""
Retriever Agent - 多模态混合检索
职责：基于Qwen3-Embedding进行向量检索，支持文本、表格、图像混合检索
"""
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentContext, AgentResult
from ..tools.vector_db import VectorDB, SearchResult


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent - 多模态混合检索
    
    功能：
    - Qwen3-Embedding 向量检索
    - 混合检索（向量 + 关键词）
    - 多模态检索（文本 + 表格 + 图像）
    - 结果重排序
    """
    
    def __init__(self, 
                 db: VectorDB = None,
                 config: Dict[str, Any] = None,
                 llm_client = None):
        super().__init__(name="retriever", config=config)
        self.db = db
        self.llm_client = llm_client
        # 增大默认检索数量，降低约束
        self.top_k = config.get("top_k", 50) if config else 50  # 从5增加到50
        self.use_rerank = config.get("rerank", True) if config else True
        self.use_hybrid = config.get("hybrid", True) if config else True
        self.min_score = config.get("min_score", 0.1) if config else 0.1  # 降低最小分数要求
    
    def run(self, context: AgentContext) -> AgentResult:
        """
        执行检索
        
        Args:
            context: Agent上下文
            
        Returns:
            包含检索证据的结果
        """
        if self.db is None:
            print(f"  [Retriever] 错误: 向量数据库未初始化")
            return AgentResult(success=False, error="向量数据库未初始化")
        
        sub_tasks = context.sub_tasks
        if not sub_tasks:
            sub_tasks = [{"query": context.question, "type": "retrieval"}]
        
        print(f"  [Retriever] 开始检索，子任务数: {len(sub_tasks)}")
        print(f"  [Retriever] 检索模式: {'混合检索' if self.use_hybrid else '向量检索'}, top_k={self.top_k}")
        
        # 对每个子任务进行检索
        all_evidences = []
        seen_keys = set()
        
        for i, task in enumerate(sub_tasks):
            query = task.get("query", task.get("task", ""))
            task_type = task.get("type", "retrieval")
            
            # 根据任务类型选择检索策略
            evidences = self._retrieve_for_task(query, task_type)
            print(f"  [Retriever] 子任务{i+1}: 检索到 {len(evidences)} 条结果")
            
            # 去重
            for ev in evidences:
                key = f"{ev['doc_id']}_{ev['page']}_{ev['text'][:50]}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    ev["task_query"] = query
                    all_evidences.append(ev)
        
        # 过滤低分证据（使用更宽松的阈值）
        before_filter = len(all_evidences)
        all_evidences = [ev for ev in all_evidences if ev.get("score", 0) >= self.min_score]
        print(f"  [Retriever] 过滤低分证据: {before_filter} -> {len(all_evidences)} (阈值={self.min_score})")
        
        # 重排序
        if self.use_rerank and len(all_evidences) > self.top_k:
            print(f"  [Retriever] 执行重排序...")
            all_evidences = self._rerank(context.question, all_evidences)
        
        # 限制返回数量（增大返回数量）
        all_evidences = all_evidences[:self.top_k * 3]  # 从2倍增加到3倍
        print(f"  [Retriever] 最终返回 {len(all_evidences)} 条证据")
        
        return AgentResult(
            success=True,
            data={"evidences": all_evidences}
        )
    
    def _retrieve_for_task(self, query: str, task_type: str) -> List[Dict[str, Any]]:
        """根据任务类型检索"""
        evidences = []
        
        if self.use_hybrid:
            # 混合检索
            results = self.db.hybrid_search(query, top_k=self.top_k)
        else:
            # 纯向量检索
            results = self.db.search(query, top_k=self.top_k)
        
        for r in results:
            evidences.append(r.to_dict())
        
        # 对于比较类任务，检索更多相关内容
        if task_type == "comparison":
            # 提取比较对象
            objects = self._extract_comparison_objects(query)
            for obj in objects:
                extra_results = self.db.search(obj, top_k=2)
                for r in extra_results:
                    evidences.append(r.to_dict())
        
        return evidences
    
    def _extract_comparison_objects(self, query: str) -> List[str]:
        """提取比较对象"""
        import re
        
        # 常见比较模式
        patterns = [
            r'(.+?)和(.+?)的(?:区别|不同|差异)',
            r'(.+?)与(.+?)的(?:区别|不同|差异)',
            r'比较(.+?)和(.+)',
            r'(.+?) vs\.? (.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return []
    
    def _rerank(self, question: str, evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序检索结果"""
        if not self.llm_client:
            # 无LLM时使用简单的分数排序
            return sorted(evidences, key=lambda x: x.get("score", 0), reverse=True)
        
        # 使用LLM进行重排序
        from ..tools.llm_client import Message
        
        # 构建重排序prompt
        evidence_texts = []
        for i, ev in enumerate(evidences[:20]):  # 限制数量
            text = ev.get("text", "")[:200]
            evidence_texts.append(f"[{i}] {text}")
        
        prompt = f"""请根据问题的相关性对以下文档片段进行排序。

问题：{question}

文档片段：
{chr(10).join(evidence_texts)}

请输出排序后的文档编号（从最相关到最不相关），格式：[0, 3, 1, 2, ...]"""
        
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm_client.chat(messages, temperature=0)
            
            # 解析排序结果
            import re
            numbers = re.findall(r'\d+', response.content)
            order = [int(n) for n in numbers if int(n) < len(evidences)]
            
            # 按新顺序排列
            reranked = []
            seen = set()
            for idx in order:
                if idx not in seen:
                    seen.add(idx)
                    reranked.append(evidences[idx])
            
            # 添加未排序的
            for i, ev in enumerate(evidences):
                if i not in seen:
                    reranked.append(ev)
            
            return reranked
        except Exception as e:
            print(f"[Warning] 重排序失败: {e}")
            return sorted(evidences, key=lambda x: x.get("score", 0), reverse=True)
    
    # 兼容旧接口
    def retrieve(self, sub_tasks: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        兼容旧接口的检索方法
        
        Args:
            sub_tasks: 子任务列表
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        self.top_k = top_k
        
        # 转换为新格式
        tasks = [{"query": t, "type": "retrieval"} for t in sub_tasks]
        
        context = AgentContext(sub_tasks=tasks)
        result = self.run(context)
        
        if result.success:
            return result.data.get("evidences", [])
        return []


class MultiModalRetriever(RetrieverAgent):
    """
    多模态检索器 - 支持文本、表格、图像的联合检索
    """
    
    def __init__(self, 
                 db: VectorDB = None,
                 config: Dict[str, Any] = None,
                 llm_client = None,
                 vl_client = None):
        super().__init__(db=db, config=config, llm_client=llm_client)
        self.vl_client = vl_client
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行多模态检索"""
        # 先执行基础检索
        result = super().run(context)
        
        if not result.success:
            return result
        
        evidences = result.data.get("evidences", [])
        
        # 对图像类型的证据进行增强
        if self.vl_client:
            evidences = self._enhance_image_evidences(evidences)
        
        return AgentResult(
            success=True,
            data={"evidences": evidences}
        )
    
    def _enhance_image_evidences(self, evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强图像证据"""
        for ev in evidences:
            if ev.get("chunk_type") == "image":
                image_path = ev.get("metadata", {}).get("image_path", "")
                if image_path:
                    try:
                        # 使用VL模型生成图像描述
                        description = self.vl_client.describe_image(image_path)
                        ev["image_description"] = description
                        ev["text"] = f"{ev.get('text', '')} {description}"
                    except Exception as e:
                        print(f"[Warning] 图像描述生成失败: {e}")
        
        return evidences
