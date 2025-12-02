"""
Reviewer Agent - 自我校验
职责：验证答案质量，决定是否需要迭代
技术：Rule-based + LLM Judge
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import BaseAgent, AgentContext, AgentResult


@dataclass
class ReviewResult:
    """审核结果"""
    final_answer: str
    confidence: float
    iterate: bool
    feedback: str = ""
    issues: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []


REVIEWER_SYSTEM_PROMPT = """你是一个严格的科学答案审核专家。请评估答案的质量。

评估维度：
1. 准确性：答案是否准确反映了证据内容
2. 完整性：答案是否完整回答了问题
3. 引用质量：引用是否准确、充分
4. 逻辑性：推理过程是否合理
5. 可验证性：答案是否可以通过证据验证

输出格式（JSON）：
{
    "scores": {
        "accuracy": 0.8,
        "completeness": 0.7,
        "citation_quality": 0.9,
        "logic": 0.85,
        "verifiability": 0.75
    },
    "overall_confidence": 0.8,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "needs_iteration": false,
    "iteration_focus": "如需迭代，重点关注的方向",
    "improved_answer": "如有改进建议，提供改进后的答案"
}"""


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent - 自我校验
    
    功能：
    - Rule-based 规则检查
    - LLM Judge 智能评估
    - 置信度计算
    - 迭代决策
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 llm_client = None):
        super().__init__(name="reviewer", config=config)
        self.llm_client = llm_client
        self.confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
        self.use_llm_judge = config.get("use_llm_judge", True) if config else True
    
    def run(self, context: AgentContext) -> AgentResult:
        """
        执行审核
        
        Args:
            context: Agent上下文
            
        Returns:
            包含审核结果的AgentResult
        """
        draft_answer = context.draft_answer
        evidences = context.evidences
        citations = context.citations
        question = context.question
        
        # 1. Rule-based 检查
        rule_result = self._rule_based_check(draft_answer, evidences, citations)
        
        # 2. LLM Judge（如果启用）
        if self.use_llm_judge and self.llm_client:
            llm_result = self._llm_judge(question, draft_answer, evidences, citations)
            # 融合结果
            final_result = self._merge_results(rule_result, llm_result)
        else:
            final_result = rule_result
        
        # 3. 决定是否需要迭代
        needs_iteration = (
            final_result.confidence < self.confidence_threshold and 
            context.iteration < context.max_iterations
        )
        final_result.iterate = needs_iteration
        
        return AgentResult(
            success=True,
            data={
                "final_answer": final_result.final_answer,
                "confidence": final_result.confidence,
                "iterate": final_result.iterate,
                "feedback": final_result.feedback,
                "issues": final_result.issues,
                "suggestions": final_result.suggestions
            }
        )
    
    def _rule_based_check(self,
                          answer: str,
                          evidences: List[Dict[str, Any]],
                          citations: List[Dict[str, Any]]) -> ReviewResult:
        """基于规则的检查"""
        issues = []
        scores = {
            "length": 0.0,
            "citation_count": 0.0,
            "evidence_coverage": 0.0,
            "no_hallucination": 0.0
        }
        
        # 1. 答案长度检查
        answer_len = len(answer)
        if answer_len < 50:
            issues.append("答案过短，可能不够完整")
            scores["length"] = 0.3
        elif answer_len < 200:
            scores["length"] = 0.6
        elif answer_len < 1000:
            scores["length"] = 0.9
        else:
            scores["length"] = 1.0
        
        # 2. 引用数量检查
        citation_pattern = r'\[来源\d+\]|\[证据\d+\]|\[\d+\]'
        citation_matches = re.findall(citation_pattern, answer)
        citation_count = len(citation_matches)
        
        if citation_count == 0:
            issues.append("答案缺少引用标记")
            scores["citation_count"] = 0.2
        elif citation_count < 2:
            issues.append("引用数量较少")
            scores["citation_count"] = 0.5
        else:
            scores["citation_count"] = min(citation_count / 5, 1.0)
        
        # 3. 证据覆盖度检查
        if evidences:
            evidence_scores = [e.get("score", 0) for e in evidences]
            avg_score = sum(evidence_scores) / len(evidence_scores)
            max_score = max(evidence_scores) if evidence_scores else 0
            
            scores["evidence_coverage"] = (avg_score + max_score) / 2
            
            if max_score < 0.3:
                issues.append("检索到的证据相关性较低")
        else:
            issues.append("没有检索到相关证据")
            scores["evidence_coverage"] = 0.0
        
        # 4. 幻觉检查（简单规则）
        hallucination_indicators = [
            "我认为", "我觉得", "可能是", "大概是",
            "据我所知", "一般来说", "通常情况下"
        ]
        hallucination_count = sum(1 for ind in hallucination_indicators if ind in answer)
        
        if hallucination_count > 2:
            issues.append("答案可能包含主观推测")
            scores["no_hallucination"] = 0.5
        elif hallucination_count > 0:
            scores["no_hallucination"] = 0.7
        else:
            scores["no_hallucination"] = 1.0
        
        # 计算总体置信度
        weights = {"length": 0.15, "citation_count": 0.25, "evidence_coverage": 0.4, "no_hallucination": 0.2}
        confidence = sum(scores[k] * weights[k] for k in scores)
        
        # 生成建议
        suggestions = []
        if scores["citation_count"] < 0.5:
            suggestions.append("增加更多引用以支持论点")
        if scores["evidence_coverage"] < 0.5:
            suggestions.append("尝试使用不同的检索策略获取更相关的证据")
        if scores["no_hallucination"] < 0.7:
            suggestions.append("减少主观推测，更多依赖证据")
        
        return ReviewResult(
            final_answer=answer,
            confidence=confidence,
            iterate=False,
            feedback=f"规则检查完成，置信度: {confidence:.2f}",
            issues=issues,
            suggestions=suggestions
        )
    
    def _llm_judge(self,
                   question: str,
                   answer: str,
                   evidences: List[Dict[str, Any]],
                   citations: List[Dict[str, Any]]) -> ReviewResult:
        """LLM评估"""
        from ..tools.llm_client import Message
        
        # 构建证据摘要
        evidence_summary = "\n".join([
            f"[证据{i+1}] {e.get('text', '')[:200]}"
            for i, e in enumerate(evidences[:5])
        ])
        
        user_prompt = f"""请评估以下答案的质量：

问题：{question}

答案：{answer}

参考证据：
{evidence_summary}

请按要求的JSON格式输出评估结果。"""
        
        messages = [
            Message(role="system", content=REVIEWER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.2)
            
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                
                scores = result.get("scores", {})
                overall_confidence = result.get("overall_confidence", 0.5)
                
                # 如果有改进的答案，使用它
                improved_answer = result.get("improved_answer", "")
                final_answer = improved_answer if improved_answer else answer
                
                return ReviewResult(
                    final_answer=final_answer,
                    confidence=overall_confidence,
                    iterate=result.get("needs_iteration", False),
                    feedback=result.get("iteration_focus", ""),
                    issues=result.get("issues", []),
                    suggestions=result.get("suggestions", [])
                )
        except Exception as e:
            print(f"[Warning] LLM评估失败: {e}")
        
        # 回退到默认结果
        return ReviewResult(
            final_answer=answer,
            confidence=0.5,
            iterate=False,
            feedback="LLM评估失败，使用默认置信度"
        )
    
    def _merge_results(self, 
                       rule_result: ReviewResult, 
                       llm_result: ReviewResult) -> ReviewResult:
        """融合规则检查和LLM评估结果"""
        # 置信度加权平均（LLM权重更高）
        merged_confidence = rule_result.confidence * 0.3 + llm_result.confidence * 0.7
        
        # 合并问题和建议
        all_issues = list(set(rule_result.issues + llm_result.issues))
        all_suggestions = list(set(rule_result.suggestions + llm_result.suggestions))
        
        # 使用LLM的改进答案（如果有）
        final_answer = llm_result.final_answer if llm_result.final_answer != rule_result.final_answer else rule_result.final_answer
        
        return ReviewResult(
            final_answer=final_answer,
            confidence=merged_confidence,
            iterate=llm_result.iterate or (merged_confidence < self.confidence_threshold),
            feedback=llm_result.feedback or rule_result.feedback,
            issues=all_issues,
            suggestions=all_suggestions
        )
    
    # 兼容旧接口
    def review(self, 
               draft: Dict[str, Any], 
               evidences: List[Dict[str, Any]]) -> ReviewResult:
        """
        兼容旧接口的审核方法
        
        Args:
            draft: 草稿答案，包含answer和citations
            evidences: 证据列表
            
        Returns:
            ReviewResult对象
        """
        context = AgentContext(
            draft_answer=draft.get("answer", ""),
            citations=draft.get("citations", []),
            evidences=evidences
        )
        result = self.run(context)
        
        if result.success:
            data = result.data
            return ReviewResult(
                final_answer=data.get("final_answer", ""),
                confidence=data.get("confidence", 0.0),
                iterate=data.get("iterate", False),
                feedback=data.get("feedback", ""),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", [])
            )
        
        return ReviewResult(
            final_answer=draft.get("answer", ""),
            confidence=0.0,
            iterate=True
        )


class IterativeReviewer(ReviewerAgent):
    """
    迭代审核器 - 支持 Reviewer → Retriever 迭代循环
    """
    
    def __init__(self,
                 config: Dict[str, Any] = None,
                 llm_client = None,
                 retriever = None):
        super().__init__(config=config, llm_client=llm_client)
        self.retriever = retriever
    
    def run_with_iteration(self, context: AgentContext) -> AgentContext:
        """
        执行带迭代的审核
        
        Args:
            context: Agent上下文
            
        Returns:
            更新后的上下文
        """
        while context.iteration < context.max_iterations:
            # 执行审核
            result = self.run(context)
            
            if not result.success:
                break
            
            data = result.data
            context.confidence = data.get("confidence", 0.0)
            context.draft_answer = data.get("final_answer", context.draft_answer)
            
            # 检查是否需要迭代
            if not data.get("iterate", False):
                break
            
            # 需要迭代：重新检索
            if self.retriever:
                context.iteration += 1
                
                # 根据反馈调整检索
                feedback = data.get("feedback", "")
                suggestions = data.get("suggestions", [])
                
                # 生成新的检索查询
                new_queries = self._generate_iteration_queries(
                    context.question, 
                    feedback, 
                    suggestions
                )
                
                # 更新子任务
                context.sub_tasks = [
                    {"query": q, "type": "retrieval"} 
                    for q in new_queries
                ]
                
                # 重新检索
                retriever_result = self.retriever.run(context)
                if retriever_result.success:
                    context.evidences = retriever_result.data.get("evidences", [])
            else:
                break
        
        return context
    
    def _generate_iteration_queries(self,
                                    question: str,
                                    feedback: str,
                                    suggestions: List[str]) -> List[str]:
        """生成迭代检索查询"""
        queries = [question]  # 保留原始问题
        
        # 基于反馈生成新查询
        if feedback:
            queries.append(f"{question} {feedback}")
        
        # 基于建议生成查询
        for suggestion in suggestions[:2]:
            if "检索" in suggestion or "证据" in suggestion:
                queries.append(f"{question} 详细")
        
        return queries[:3]  # 限制查询数量
