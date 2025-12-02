"""
Reflector Agent - 自我反思
基于对话历史进行反思，提供改进建议
"""
import re
import json
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentContext, AgentResult
from ..models.memory import Reflection, Conversation, ConversationTurn


REFLECTION_PROMPT = """你是一个专业的对话质量分析专家。请分析以下对话历史，进行深度自我反思。

分析维度：
1. 回答质量：答案是否准确、完整、有帮助
2. 证据使用：是否充分利用了检索到的证据
3. 用户需求：是否真正理解并满足了用户需求
4. 改进空间：有哪些可以改进的地方
5. 关键洞察：从对话中提取的重要发现
6. 知识空白：用户可能需要补充的知识领域
7. 用户偏好：用户的提问风格、关注点等
8. 后续建议：基于当前对话，用户可能感兴趣的后续问题

输出格式（JSON）：
{
    "overall_assessment": "总体评估（详细描述对话质量、用户满意度预测等）",
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"],
    "suggestions": [
        "建议用户提供更具体的问题描述",
        "建议上传更多相关文档"
    ],
    "improvement_areas": [
        "需要更多关于XX领域的文档",
        "问题表述可以更加具体"
    ],
    "key_insights": [
        "用户对XX领域有深入研究需求",
        "文档中关于YY的内容最受关注"
    ],
    "knowledge_gaps": [
        "缺少关于XX方法的详细文档",
        "需要补充YY领域的背景知识"
    ],
    "user_preferences": {
        "question_style": "技术性/概念性/应用性",
        "detail_level": "详细/简洁",
        "focus_areas": ["领域1", "领域2"]
    },
    "topic_summary": "本次对话主要围绕XX主题展开，涉及YY和ZZ等方面",
    "follow_up_questions": [
        "基于当前讨论，您可能想了解...",
        "进一步探索的方向可以是..."
    ],
    "confidence_trend": "置信度趋势分析",
    "memory_for_next": "下次对话时应该记住的关键信息"
}"""


class ReflectorAgent(BaseAgent):
    """
    反思Agent - 基于对话历史进行自我反思
    
    功能：
    - 分析对话质量
    - 识别改进空间
    - 提供用户建议
    - 生成反思报告
    """
    
    def __init__(self,
                 config: Dict[str, Any] = None,
                 llm_client = None):
        super().__init__(name="reflector", config=config)
        self.llm_client = llm_client
        self.min_turns_for_reflection = config.get("min_turns", 1) if config else 1
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行反思（基于AgentContext）"""
        # 从metadata中获取对话信息
        conversation = context.metadata.get("conversation")
        if not conversation:
            return AgentResult(
                success=False,
                error="缺少对话历史"
            )
        
        reflection = self.reflect_on_conversation(conversation)
        return AgentResult(
            success=True,
            data={"reflection": reflection}
        )
    
    def reflect_on_conversation(self, conversation: Conversation) -> Reflection:
        """
        对对话进行反思
        
        Args:
            conversation: 对话对象
            
        Returns:
            反思结果
        """
        if len(conversation.turns) < self.min_turns_for_reflection:
            return self._create_minimal_reflection(conversation)
        
        if self.llm_client:
            return self._reflect_with_llm(conversation)
        else:
            return self._reflect_simple(conversation)
    
    def _reflect_with_llm(self, conversation: Conversation) -> Reflection:
        """使用LLM进行详细反思"""
        from ..tools.llm_client import Message
        
        # 构建对话历史
        history = self._build_conversation_history(conversation)
        
        # 获取之前的反思（如果有）用于连续性
        previous_reflection = ""
        if conversation.reflection:
            previous_reflection = f"\n\n之前的反思摘要：\n{conversation.reflection.content[:500]}"
        
        user_prompt = f"""请分析以下对话历史并进行深度反思：

{history}
{previous_reflection}

请按要求的JSON格式输出详细的反思结果，特别注意提取可用于下次对话的关键信息。"""
        
        messages = [
            Message(role="system", content=REFLECTION_PROMPT),
            Message(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                
                return Reflection(
                    reflection_id=secrets.token_hex(16),
                    conversation_id=conversation.conversation_id,
                    content=result.get("overall_assessment", ""),
                    suggestions=result.get("suggestions", []),
                    improvement_areas=result.get("improvement_areas", []),
                    key_insights=result.get("key_insights", []),
                    knowledge_gaps=result.get("knowledge_gaps", []),
                    user_preferences=result.get("user_preferences", {}),
                    topic_summary=result.get("topic_summary", ""),
                    follow_up_questions=result.get("follow_up_questions", [])
                )
        except Exception as e:
            print(f"[Warning] LLM反思失败: {e}")
        
        return self._reflect_simple(conversation)
    
    def _reflect_simple(self, conversation: Conversation) -> Reflection:
        """简单的反思（无LLM）- 更详细版本"""
        suggestions = []
        improvement_areas = []
        key_insights = []
        knowledge_gaps = []
        follow_up_questions = []
        
        # 分析对话
        total_confidence = 0
        low_confidence_count = 0
        topics = []
        
        for turn in conversation.turns:
            confidence = turn.confidence
            total_confidence += confidence
            
            if confidence < 0.5:
                low_confidence_count += 1
            
            # 提取话题关键词
            question_words = turn.question[:100]
            topics.append(question_words)
        
        avg_confidence = total_confidence / len(conversation.turns) if conversation.turns else 0
        
        # 生成建议
        if avg_confidence < 0.6:
            suggestions.append("建议提供更多相关文档以提高回答质量")
            improvement_areas.append("证据覆盖度不足")
            knowledge_gaps.append("当前文档可能未覆盖问题的核心领域")
        
        if low_confidence_count > len(conversation.turns) / 2:
            suggestions.append("建议将问题拆分为更具体的子问题")
            improvement_areas.append("问题可能过于宽泛")
        
        # 检查是否有重新生成
        regenerate_count = sum(len(t.regenerated_answers) for t in conversation.turns)
        if regenerate_count > 0:
            suggestions.append("您多次重新生成了答案，建议提供更明确的问题描述")
            key_insights.append("用户对回答质量有较高要求")
        
        # 生成后续问题建议
        if conversation.turns:
            last_question = conversation.turns[-1].question
            follow_up_questions.append(f"关于'{last_question[:30]}...'的更深入探讨")
            follow_up_questions.append("相关领域的最新研究进展")
        
        # 生成话题总结
        topic_summary = f"本次对话共{len(conversation.turns)}轮，主要围绕用户提出的问题展开讨论"
        
        # 生成反思内容
        content = f"""对话分析报告：
- 对话轮次：{len(conversation.turns)}
- 平均置信度：{avg_confidence:.1%}
- 低置信度回答：{low_confidence_count} 次
- 重新生成次数：{regenerate_count} 次

总体评估：{"回答质量较好，系统能够有效利用文档内容回答问题" if avg_confidence >= 0.6 else "回答质量有待提高，建议补充更多文档或细化问题"}

用户交互特点：{"用户倾向于深入探讨问题" if regenerate_count > 0 else "用户对初次回答较为满意"}"""
        
        return Reflection(
            reflection_id=secrets.token_hex(16),
            conversation_id=conversation.conversation_id,
            content=content,
            suggestions=suggestions if suggestions else ["继续保持当前的提问方式"],
            improvement_areas=improvement_areas if improvement_areas else ["暂无明显改进空间"],
            key_insights=key_insights if key_insights else ["用户正在探索文档内容"],
            knowledge_gaps=knowledge_gaps,
            user_preferences={"detail_level": "standard", "focus_areas": []},
            topic_summary=topic_summary,
            follow_up_questions=follow_up_questions
        )
    
    def _create_minimal_reflection(self, conversation: Conversation) -> Reflection:
        """创建最小反思（对话轮次不足时）"""
        return Reflection(
            reflection_id=secrets.token_hex(16),
            conversation_id=conversation.conversation_id,
            content="对话刚刚开始，暂无足够信息进行深度反思。",
            suggestions=["继续提问以获得更多反馈"],
            improvement_areas=[]
        )
    
    def _build_conversation_history(self, conversation: Conversation) -> str:
        """构建对话历史文本"""
        history_parts = []
        
        for i, turn in enumerate(conversation.turns):
            history_parts.append(f"--- 轮次 {i+1} ---")
            history_parts.append(f"用户问题：{turn.question}")
            history_parts.append(f"系统回答：{turn.answer[:500]}...")
            history_parts.append(f"置信度：{turn.confidence:.1%}")
            history_parts.append(f"引用数量：{len(turn.citations)}")
            if turn.regenerated_answers:
                history_parts.append(f"重新生成次数：{len(turn.regenerated_answers)}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def get_quick_feedback(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        获取单轮对话的快速反馈
        
        Args:
            turn: 对话轮次
            
        Returns:
            快速反馈字典
        """
        feedback = {
            "confidence_level": "high" if turn.confidence >= 0.7 else "medium" if turn.confidence >= 0.4 else "low",
            "has_citations": len(turn.citations) > 0,
            "citation_count": len(turn.citations),
            "suggestions": []
        }
        
        if turn.confidence < 0.5:
            feedback["suggestions"].append("回答置信度较低，建议补充相关文档")
        
        if len(turn.citations) == 0:
            feedback["suggestions"].append("回答缺少引用，可能需要更多证据支持")
        
        if not feedback["suggestions"]:
            feedback["suggestions"].append("回答质量良好")
        
        return feedback
