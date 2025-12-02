"""
DeepThinker Agent - 深度思考模式
类似DeepSeek-R1的推理思路展示，支持多文档联合推理追踪
"""
import re
import json
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator

from .base import BaseAgent, AgentContext, AgentResult
from ..models.memory import ThinkingStep, ReasoningTrace


DEEP_THINKING_PROMPT = """你是一个专业的科学研究助手，需要展示完整的推理过程。

请按照以下步骤进行深度思考：

1. **问题理解**：分析用户问题的核心需求
2. **证据分析**：逐条分析每个证据的相关性和可信度
3. **跨文档关联**：找出不同文档之间的联系和矛盾
4. **推理链构建**：建立从证据到结论的逻辑链
5. **置信度评估**：评估答案的可靠性
6. **答案生成**：生成最终答案

**重要：数学公式格式要求**
- 行内公式使用单个美元符号包裹，如：$E=mc^2$
- 块级公式（独立成行的公式）使用双美元符号包裹，如：
  $$\\text{Attn}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
- 所有数学符号、变量、方程都必须使用LaTeX格式
- 希腊字母如 $\\alpha$, $\\beta$, $\\gamma$ 等使用LaTeX命令
- 分数使用 $\\frac{a}{b}$，根号使用 $\\sqrt{x}$
- 矩阵、向量等使用适当的LaTeX环境

输出格式（JSON）：
{
    "thinking_steps": [
        {
            "step_id": 1,
            "title": "问题理解",
            "content": "详细分析...",
            "sources": []
        },
        {
            "step_id": 2,
            "title": "证据分析",
            "content": "证据1来自[文档A, p.5]，说明了...",
            "sources": [{"doc": "文档A", "page": 5, "quote": "引用内容"}]
        }
    ],
    "document_sources": [
        {"doc_id": "xxx", "source": "文档名", "pages": [1,2,3], "contribution": "该文档提供了..."}
    ],
    "reasoning_chain": "从证据A推导出B，结合证据C得出结论D",
    "answer": "最终答案（数学公式使用LaTeX格式，如 $x^2$ 或 $$\\sum_{i=1}^n x_i$$）",
    "confidence": 0.85,
    "calibration": {
        "evidence_strength": 0.8,
        "source_reliability": 0.9,
        "reasoning_validity": 0.85,
        "coverage_completeness": 0.7
    }
}"""


class DeepThinkerAgent(BaseAgent):
    """
    深度思考Agent - 展示完整推理过程
    
    功能：
    - 展示类似DeepSeek-R1的深度思考过程
    - 多文档联合推理追踪
    - 置信度自校准
    - 支持流式输出思考过程
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 llm_client = None):
        super().__init__(name="deep_thinker", config=config)
        self.llm_client = llm_client
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行深度思考"""
        question = context.question
        evidences = context.evidences
        captions = context.captions
        
        print(f"  [DeepThinker] 开始深度思考，证据数: {len(evidences) if evidences else 0}")
        
        if not evidences:
            print(f"  [DeepThinker] 未找到证据，返回空结果")
            return AgentResult(
                success=True,
                data={
                    "reasoning_trace": self._create_empty_trace(question),
                    "answer": "未找到相关证据，无法进行深度分析。",
                    "confidence": 0.0
                }
            )
        
        # 执行深度思考
        if self.llm_client:
            print(f"  [DeepThinker] 使用LLM进行深度推理")
            result = self._deep_think_with_llm(question, evidences, captions)
        else:
            print(f"  [DeepThinker] 使用简单推理模式")
            result = self._deep_think_simple(question, evidences, captions)
        
        print(f"  [DeepThinker] 推理完成，置信度: {result.get('confidence', 0):.2%}")
        return AgentResult(success=True, data=result)
    
    def _deep_think_with_llm(self,
                             question: str,
                             evidences: List[Dict[str, Any]],
                             captions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM进行深度思考"""
        from ..tools.llm_client import Message
        
        # 构建证据上下文
        evidence_context = self._build_evidence_context(evidences, captions)
        
        user_prompt = f"""问题：{question}

证据材料：
{evidence_context}

请进行深度思考分析，展示完整的推理过程。"""
        
        messages = [
            Message(role="system", content=DEEP_THINKING_PROMPT),
            Message(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3, max_tokens=4096)
            
            # 解析JSON响应
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                json_str = json_match.group()
                # 修复LaTeX转义字符问题：将单反斜杠转为双反斜杠（排除已有的双反斜杠和有效JSON转义）
                # 匹配 \ 后面不是 n, r, t, b, f, ", \, / 或 u 的情况
                json_str = re.sub(r'\\(?![nrtbf"\\\/u])', r'\\\\', json_str)
                result = json.loads(json_str)
                
                # 构建推理追踪
                trace = self._build_reasoning_trace(question, result, evidences)
                
                return {
                    "reasoning_trace": trace,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5),
                    "calibration_details": result.get("calibration", {})
                }
        except Exception as e:
            print(f"[Warning] 深度思考失败: {e}")
        
        return self._deep_think_simple(question, evidences, captions)
    
    def _deep_think_simple(self,
                           question: str,
                           evidences: List[Dict[str, Any]],
                           captions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简单的深度思考（无LLM）"""
        trace_id = secrets.token_hex(16)
        
        # 构建思考步骤
        thinking_steps = []
        
        # Step 1: 问题理解
        thinking_steps.append(ThinkingStep(
            step_id=1,
            title="问题理解",
            content=f"用户询问：「{question}」\n需要从文献中找到相关信息进行回答。"
        ))
        
        # Step 2: 证据分析
        evidence_analysis = []
        document_sources = []
        seen_docs = set()
        
        for i, ev in enumerate(evidences[:10]):
            source = ev.get("source", "未知来源")
            page = ev.get("page", 0)
            text = ev.get("text", "")[:200]
            score = ev.get("score", 0)
            # 确保 doc_id 不为空字符串
            doc_id = ev.get("doc_id") or f"doc_{i}"
            
            evidence_analysis.append(
                f"- 证据{i+1} [来源: {source}, p.{page}, 相关度: {score:.2f}]\n  内容: {text}..."
            )
            
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                document_sources.append({
                    "doc_id": doc_id,
                    "source": source,
                    "pages": [page],
                    "contribution": f"提供了关于问题的相关信息",
                    "relevance_score": score
                })
        
        thinking_steps.append(ThinkingStep(
            step_id=2,
            title="证据分析",
            content="\n".join(evidence_analysis),
            sources=[{"doc": ev.get("source", ""), "page": ev.get("page", 0)} 
                    for ev in evidences[:5]]
        ))
        
        # Step 3: 跨文档关联
        if len(seen_docs) > 1:
            cross_doc_content = f"共检索到 {len(seen_docs)} 个不同文档的相关内容，需要综合分析。"
        else:
            cross_doc_content = "主要证据来自单一文档。"
        
        thinking_steps.append(ThinkingStep(
            step_id=3,
            title="跨文档关联",
            content=cross_doc_content
        ))
        
        # Step 4: 推理链构建
        thinking_steps.append(ThinkingStep(
            step_id=4,
            title="推理链构建",
            content="基于检索到的证据，综合各文档信息进行推理。"
        ))
        
        # Step 5: 置信度评估
        scores = [ev.get("score", 0) for ev in evidences]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        calibration = {
            "evidence_strength": avg_score,
            "source_reliability": 0.7,
            "reasoning_validity": 0.6,
            "coverage_completeness": min(len(evidences) / 5, 1.0)
        }
        confidence = sum(calibration.values()) / len(calibration)
        
        thinking_steps.append(ThinkingStep(
            step_id=5,
            title="置信度评估",
            content=f"证据强度: {calibration['evidence_strength']:.2f}\n"
                   f"来源可靠性: {calibration['source_reliability']:.2f}\n"
                   f"推理有效性: {calibration['reasoning_validity']:.2f}\n"
                   f"覆盖完整性: {calibration['coverage_completeness']:.2f}\n"
                   f"综合置信度: {confidence:.2f}"
        ))
        
        # 生成答案
        answer_parts = [f"根据检索到的 {len(evidences)} 条证据，关于「{question}」的分析如下：\n"]
        for i, ev in enumerate(evidences[:5]):
            source = ev.get("source", "未知")
            page = ev.get("page", "?")
            text = ev.get("text", "")[:300]
            answer_parts.append(f"[来源{i+1}: {source}, p.{page}] {text}")
        
        answer = "\n\n".join(answer_parts)
        
        # 构建推理追踪
        trace = ReasoningTrace(
            trace_id=trace_id,
            question=question,
            thinking_steps=thinking_steps,
            document_sources=document_sources,
            final_answer=answer,
            confidence=confidence,
            calibration_details=calibration
        )
        
        return {
            "reasoning_trace": trace,
            "answer": answer,
            "confidence": confidence,
            "calibration_details": calibration
        }
    
    def _build_evidence_context(self,
                                evidences: List[Dict[str, Any]],
                                captions: List[Dict[str, Any]]) -> str:
        """构建证据上下文"""
        context_parts = []
        
        for i, ev in enumerate(evidences[:10]):
            source = ev.get("source", "未知来源")
            page = ev.get("page", "?")
            text = ev.get("text", "")[:500]
            doc_id = ev.get("doc_id", "")
            
            context_parts.append(
                f"[证据{i+1}] 文档: {source} | 页码: {page} | ID: {doc_id}\n{text}"
            )
        
        for i, cap in enumerate(captions[:5]):
            caption_text = cap.get("generated_caption", cap.get("caption", ""))
            if caption_text:
                page = cap.get("page", "?")
                context_parts.append(f"[图像{i+1}] 页码: {page}\n{caption_text[:300]}")
        
        return "\n\n".join(context_parts)
    
    def _build_reasoning_trace(self,
                               question: str,
                               result: Dict[str, Any],
                               evidences: List[Dict[str, Any]]) -> ReasoningTrace:
        """构建推理追踪对象"""
        trace_id = secrets.token_hex(16)
        
        # 转换思考步骤
        thinking_steps = []
        for step_data in result.get("thinking_steps", []):
            thinking_steps.append(ThinkingStep(
                step_id=step_data.get("step_id", 0),
                title=step_data.get("title", ""),
                content=step_data.get("content", ""),
                sources=step_data.get("sources", [])
            ))
        
        # 文档来源
        document_sources = result.get("document_sources", [])
        if not document_sources:
            # 从证据中提取
            seen_docs = {}
            for i, ev in enumerate(evidences):
                # 确保 doc_id 不为空字符串
                doc_id = ev.get("doc_id") or f"evidence_{i}"
                if doc_id not in seen_docs:
                    seen_docs[doc_id] = {
                        "doc_id": doc_id,
                        "source": ev.get("source", ""),
                        "pages": [ev.get("page", 0)],
                        "contribution": "提供相关证据"
                    }
                else:
                    page = ev.get("page", 0)
                    if page not in seen_docs[doc_id]["pages"]:
                        seen_docs[doc_id]["pages"].append(page)
            document_sources = list(seen_docs.values())
        
        return ReasoningTrace(
            trace_id=trace_id,
            question=question,
            thinking_steps=thinking_steps,
            document_sources=document_sources,
            final_answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.5),
            calibration_details=result.get("calibration", {})
        )
    
    def _create_empty_trace(self, question: str) -> ReasoningTrace:
        """创建空的推理追踪"""
        return ReasoningTrace(
            trace_id=secrets.token_hex(16),
            question=question,
            thinking_steps=[
                ThinkingStep(
                    step_id=1,
                    title="问题理解",
                    content=f"用户询问：「{question}」"
                ),
                ThinkingStep(
                    step_id=2,
                    title="证据检索",
                    content="未找到相关证据"
                )
            ],
            document_sources=[],
            final_answer="",
            confidence=0.0
        )
    
    def think_stream(self, 
                     question: str,
                     evidences: List[Dict[str, Any]],
                     captions: List[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        流式输出思考过程 - 类似DeepSeek的连续思考文字流
        
        Yields:
            思考内容字典，包含连续的思考文字
        """
        import time
        captions = captions or []
        start_time = time.time()
        
        # 开始思考
        yield {"type": "thinking_start", "message": "开始深度思考..."}
        
        # 收集文档信息
        unique_docs = {}
        for i, ev in enumerate(evidences):
            doc_id = ev.get("doc_id", f"doc_{i}")
            source = ev.get("source", "未知")
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {"source": source, "pages": [], "scores": [], "texts": []}
            unique_docs[doc_id]["pages"].append(ev.get("page", 0))
            unique_docs[doc_id]["scores"].append(ev.get("score", 0))
            unique_docs[doc_id]["texts"].append(ev.get("text", "")[:200])
        
        # 生成连续的思考文字
        thinking_text = ""
        
        # 第一段：问题分析
        thinking_text += f"好的，用户现在让我分析关于「{question}」的问题。"
        thinking_text += f"首先，我需要理解这个问题的核心需求。"
        yield {"type": "thinking_delta", "content": thinking_text}
        
        # 分析问题类型
        keywords = self._extract_keywords(question)
        thinking_text = f"\n\n用户的问题涉及到「{keywords}」这些关键概念。"
        thinking_text += "这是一个需要从文献中检索相关信息来回答的问题。"
        thinking_text += f"我检索到了 {len(evidences)} 条相关证据，来自 {len(unique_docs)} 个不同的文档。"
        yield {"type": "thinking_delta", "content": thinking_text}
        
        # 第二段：证据分析
        thinking_text = "\n\n接下来，我需要分析这些证据的质量和相关性。"
        yield {"type": "thinking_delta", "content": thinking_text}
        
        for i, ev in enumerate(evidences[:5]):
            source = ev.get("source", "未知")
            page = ev.get("page", "?")
            score = ev.get("score", 0)
            text = ev.get("text", "")[:150]
            
            thinking_text = f"\n\n证据{i+1}来自「{source}」第{page}页，相关度为{score:.2f}。"
            if score > 0.7:
                thinking_text += "这条证据高度相关，可以作为主要支撑。"
            elif score > 0.4:
                thinking_text += "这条证据中等相关，可以作为辅助参考。"
            else:
                thinking_text += "这条证据相关性较低，需要谨慎使用。"
            
            if text:
                thinking_text += f"内容摘要：「{text}...」"
            yield {"type": "thinking_delta", "content": thinking_text}
        
        # 第三段：跨文档分析
        if len(unique_docs) > 1:
            thinking_text = f"\n\n现在我需要分析这 {len(unique_docs)} 个文档之间的关系。"
            thinking_text += "多个文档从不同角度提供了相关信息，我需要综合各文档的观点。"
            yield {"type": "thinking_delta", "content": thinking_text}
            
            for doc_id, info in list(unique_docs.items())[:3]:
                avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
                pages = sorted(set(info["pages"]))[:5]
                thinking_text = f"\n\n「{info['source']}」涉及第{', '.join(map(str, pages))}页，"
                thinking_text += f"平均相关度{avg_score:.2f}，"
                thinking_text += f"{'是主要信息来源' if avg_score > 0.6 else '提供辅助信息'}。"
                yield {"type": "thinking_delta", "content": thinking_text}
        else:
            thinking_text = "\n\n注意到证据主要来自单一文档，答案可能存在视角局限性。"
            yield {"type": "thinking_delta", "content": thinking_text}
        
        # 第四段：推理过程
        thinking_text = "\n\n现在开始综合这些证据进行推理。"
        thinking_text += f"基于 {len(evidences)} 条证据，我需要建立从证据到结论的逻辑链。"
        yield {"type": "thinking_delta", "content": thinking_text}
        
        # 执行实际推理
        context = AgentContext(
            question=question,
            evidences=evidences,
            captions=captions
        )
        result = self.run(context)
        
        if result.success:
            data = result.data
            confidence = data.get("confidence", 0.5)
            calibration = data.get("calibration_details", {})
            
            # 推理结论
            thinking_text = "\n\n通过分析各文档的信息，我发现："
            
            # 提取关键发现
            if evidences:
                top_evidence = evidences[0]
                thinking_text += f"\n• 最相关的证据来自「{top_evidence.get('source', '未知')}」"
                thinking_text += f"第{top_evidence.get('page', '?')}页"
            
            yield {"type": "thinking_delta", "content": thinking_text}
            
            # 第五段：置信度评估
            thinking_text = f"\n\n最后，我需要评估答案的可信度。"
            thinking_text += f"综合置信度为 {confidence:.1%}。"
            
            if calibration:
                dimension_names = {
                    "evidence_strength": "证据强度",
                    "source_reliability": "来源可靠性",
                    "reasoning_validity": "推理有效性",
                    "coverage_completeness": "覆盖完整性"
                }
                for k, v in calibration.items():
                    name = dimension_names.get(k, k)
                    thinking_text += f"\n• {name}：{v:.2f}"
            
            if confidence >= 0.7:
                thinking_text += "\n\n答案可信度较高，证据充分且推理合理。"
            elif confidence >= 0.4:
                thinking_text += "\n\n答案可信度中等，建议参考原文进一步确认。"
            else:
                thinking_text += "\n\n答案可信度较低，证据不足，请谨慎参考。"
            
            yield {"type": "thinking_delta", "content": thinking_text}
            
            # 计算思考时间
            elapsed_time = time.time() - start_time
            
            # 思考完成
            yield {
                "type": "thinking_complete",
                "elapsed_seconds": round(elapsed_time, 1)
            }
            
            # 最终结果
            trace = data.get("reasoning_trace")
            yield {
                "type": "final_result",
                "reasoning_trace": trace.to_dict() if hasattr(trace, "to_dict") else trace,
                "answer": data.get("answer", ""),
                "confidence": confidence,
                "calibration_details": calibration
            }
    
    def _extract_keywords(self, question: str) -> str:
        """从问题中提取关键词"""
        stop_words = {'的', '是', '在', '有', '和', '与', '了', '什么', '如何', '为什么', '怎么', '哪些', '哪个', '请', '问'}
        
        # 对于英文，按空格分词
        english_words = re.findall(r'[a-zA-Z]+', question)
        
        # 组合关键词
        keywords = [w for w in question.split() if w not in stop_words and len(w) > 1]
        keywords.extend(english_words)
        
        return '、'.join(keywords[:5]) if keywords else question[:20]
