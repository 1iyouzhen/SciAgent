"""
记忆库系统 - 保存对话历史，支持相似问题检索和自我反思
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np


@dataclass
class ThinkingStep:
    """推理步骤"""
    step_id: int
    title: str
    content: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningTrace:
    """推理追踪 - 记录完整的推理过程"""
    trace_id: str
    question: str
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    document_sources: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    calibration_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "thinking_steps": [asdict(s) for s in self.thinking_steps],
            "document_sources": self.document_sources,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "calibration_details": self.calibration_details
        }


@dataclass
class Reflection:
    """自我反思 - 更详细的反思，可用于下次对话"""
    reflection_id: str
    conversation_id: str
    content: str
    suggestions: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    # 新增：详细反思字段
    key_insights: List[str] = field(default_factory=list)  # 关键洞察
    knowledge_gaps: List[str] = field(default_factory=list)  # 知识空白
    user_preferences: Dict[str, Any] = field(default_factory=dict)  # 用户偏好
    topic_summary: str = ""  # 话题总结
    follow_up_questions: List[str] = field(default_factory=list)  # 后续问题建议
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedReport:
    """生成的报告记录"""
    report_id: str
    conversation_id: str
    user_id: str
    requirement: str  # 用户需求
    report: Dict[str, Any]  # 报告内容
    citations: List[Dict[str, Any]]  # 引用列表
    confidence: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "requirement": self.requirement,
            "report": self.report,
            "citations": self.citations,
            "confidence": self.confidence,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedReport":
        return cls(
            report_id=data["report_id"],
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            requirement=data.get("requirement", ""),
            report=data.get("report", {}),
            citations=data.get("citations", []),
            confidence=data.get("confidence", 0.0),
            created_at=data.get("created_at", "")
        )


@dataclass
class ConversationTurn:
    """对话轮次"""
    turn_id: str
    question: str
    answer: str
    reasoning_trace: Optional[ReasoningTrace] = None
    confidence: float = 0.0
    citations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    regenerated_answers: List[Dict[str, Any]] = field(default_factory=list)  # 重新生成的答案历史
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle reasoning_trace being either a ReasoningTrace object or already a dict
        if self.reasoning_trace is None:
            trace_dict = None
        elif isinstance(self.reasoning_trace, dict):
            trace_dict = self.reasoning_trace
        else:
            trace_dict = self.reasoning_trace.to_dict()
        
        return {
            "turn_id": self.turn_id,
            "question": self.question,
            "answer": self.answer,
            "reasoning_trace": trace_dict,
            "confidence": self.confidence,
            "citations": self.citations,
            "timestamp": self.timestamp,
            "regenerated_answers": self.regenerated_answers
        }


@dataclass
class Conversation:
    """对话会话"""
    conversation_id: str
    user_id: str
    title: str = ""
    turns: List[ConversationTurn] = field(default_factory=list)
    reflection: Optional[Reflection] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "turns": [t.to_dict() for t in self.turns],
            "reflection": self.reflection.to_dict() if self.reflection else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        turns = []
        for t in data.get("turns", []):
            trace = None
            if t.get("reasoning_trace"):
                rt = t["reasoning_trace"]
                steps = [ThinkingStep(**s) for s in rt.get("thinking_steps", [])]
                trace = ReasoningTrace(
                    trace_id=rt["trace_id"],
                    question=rt["question"],
                    thinking_steps=steps,
                    document_sources=rt.get("document_sources", []),
                    final_answer=rt.get("final_answer", ""),
                    confidence=rt.get("confidence", 0.0),
                    calibration_details=rt.get("calibration_details", {})
                )
            turns.append(ConversationTurn(
                turn_id=t["turn_id"],
                question=t["question"],
                answer=t["answer"],
                reasoning_trace=trace,
                confidence=t.get("confidence", 0.0),
                citations=t.get("citations", []),
                timestamp=t.get("timestamp", ""),
                regenerated_answers=t.get("regenerated_answers", [])
            ))
        
        reflection = None
        if data.get("reflection"):
            ref_data = data["reflection"]
            reflection = Reflection(
                reflection_id=ref_data.get("reflection_id", ""),
                conversation_id=ref_data.get("conversation_id", ""),
                content=ref_data.get("content", ""),
                suggestions=ref_data.get("suggestions", []),
                improvement_areas=ref_data.get("improvement_areas", []),
                key_insights=ref_data.get("key_insights", []),
                knowledge_gaps=ref_data.get("knowledge_gaps", []),
                user_preferences=ref_data.get("user_preferences", {}),
                topic_summary=ref_data.get("topic_summary", ""),
                follow_up_questions=ref_data.get("follow_up_questions", []),
                created_at=ref_data.get("created_at", "")
            )
        
        return cls(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            title=data.get("title", ""),
            turns=turns,
            reflection=reflection,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", "")
        )


class MemoryStore:
    """记忆库存储"""
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conversations: Dict[str, Conversation] = {}
        self._reports: Dict[str, GeneratedReport] = {}  # 报告存储
        self._question_embeddings: Dict[str, np.ndarray] = {}
        self._encoder = None
        self._load_all()
    
    def _get_encoder(self):
        """获取编码器（延迟加载）"""
        if self._encoder is None:
            try:
                from ..tools.vector_db import QwenEmbedding
                self._encoder = QwenEmbedding()
            except:
                self._encoder = "fallback"
        return self._encoder
    
    def _load_all(self):
        """加载所有对话和报告"""
        for file_path in self.data_dir.glob("*.json"):
            try:
                # 跳过空文件
                if file_path.stat().st_size == 0:
                    print(f"[Warning] 跳过空文件: {file_path}")
                    continue
                    
                if file_path.name.startswith("conv_"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        conv = Conversation.from_dict(data)
                        self._conversations[conv.conversation_id] = conv
                elif file_path.name.startswith("report_"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        report = GeneratedReport.from_dict(data)
                        self._reports[report.report_id] = report
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON解析失败，跳过文件 {file_path}: {e}")
            except Exception as e:
                print(f"[Warning] 加载文件失败 {file_path}: {e}")
    
    def _save_conversation(self, conv: Conversation):
        """保存对话"""
        file_path = self.data_dir / f"conv_{conv.conversation_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
    
    def create_conversation(self, user_id: str, title: str = "") -> Conversation:
        """创建新对话"""
        import secrets
        conv_id = secrets.token_hex(16)
        conv = Conversation(
            conversation_id=conv_id,
            user_id=user_id,
            title=title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        self._conversations[conv_id] = conv
        self._save_conversation(conv)
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self._conversations.get(conversation_id)
    
    def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """获取用户的所有对话"""
        return [c for c in self._conversations.values() if c.user_id == user_id]
    
    def add_turn(self, conversation_id: str, turn: ConversationTurn) -> bool:
        """添加对话轮次"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return False
        
        conv.turns.append(turn)
        conv.updated_at = datetime.now().isoformat()
        
        # 更新标题（使用第一个问题）
        if not conv.title or conv.title.startswith("对话 "):
            conv.title = turn.question[:50] + ("..." if len(turn.question) > 50 else "")
        
        self._save_conversation(conv)
        return True
    
    def update_turn_answer(self, conversation_id: str, turn_id: str, 
                           new_answer: str, new_trace: ReasoningTrace = None,
                           new_confidence: float = 0.0) -> bool:
        """更新轮次答案（重新生成）"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return False
        
        for turn in conv.turns:
            if turn.turn_id == turn_id:
                # 保存旧答案到历史
                # Handle reasoning_trace being either a ReasoningTrace object or already a dict
                if turn.reasoning_trace is None:
                    old_trace = None
                elif isinstance(turn.reasoning_trace, dict):
                    old_trace = turn.reasoning_trace
                else:
                    old_trace = turn.reasoning_trace.to_dict()
                
                turn.regenerated_answers.append({
                    "answer": turn.answer,
                    "confidence": turn.confidence,
                    "reasoning_trace": old_trace,
                    "timestamp": datetime.now().isoformat()
                })
                # 更新新答案
                turn.answer = new_answer
                turn.confidence = new_confidence
                if new_trace:
                    turn.reasoning_trace = new_trace
                
                conv.updated_at = datetime.now().isoformat()
                self._save_conversation(conv)
                return True
        
        return False
    
    def add_reflection(self, conversation_id: str, reflection: Reflection) -> bool:
        """添加反思"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return False
        
        conv.reflection = reflection
        conv.updated_at = datetime.now().isoformat()
        self._save_conversation(conv)
        return True
    
    def find_similar_questions(self, question: str, user_id: str, 
                               top_k: int = 3) -> List[Tuple[ConversationTurn, float]]:
        """查找相似问题"""
        encoder = self._get_encoder()
        
        if encoder == "fallback":
            # 简单的关键词匹配
            return self._keyword_search(question, user_id, top_k)
        
        # 向量相似度搜索
        query_vec = encoder.encode([question])[0]
        
        results = []
        for conv in self._conversations.values():
            if conv.user_id != user_id:
                continue
            
            for turn in conv.turns:
                # 计算相似度
                turn_vec = encoder.encode([turn.question])[0]
                similarity = np.dot(query_vec, turn_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(turn_vec) + 1e-9
                )
                results.append((turn, float(similarity)))
        
        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _keyword_search(self, question: str, user_id: str, 
                        top_k: int) -> List[Tuple[ConversationTurn, float]]:
        """关键词搜索回退"""
        import re
        query_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', question.lower()))
        
        results = []
        for conv in self._conversations.values():
            if conv.user_id != user_id:
                continue
            
            for turn in conv.turns:
                turn_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', turn.question.lower()))
                intersection = len(query_tokens & turn_tokens)
                union = len(query_tokens | turn_tokens)
                similarity = intersection / union if union > 0 else 0
                results.append((turn, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """删除对话"""
        if conversation_id not in self._conversations:
            return False
        
        del self._conversations[conversation_id]
        file_path = self.data_dir / f"conv_{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True
    
    def get_user_history_context(self, user_id: str, 
                                  current_question: str,
                                  max_turns: int = 10) -> Dict[str, Any]:
        """
        获取用户的历史上下文（用于跨对话记忆）
        
        Args:
            user_id: 用户ID
            current_question: 当前问题
            max_turns: 最大返回轮次
            
        Returns:
            包含历史对话和相似问题的上下文
        """
        # 获取所有对话
        conversations = self.get_user_conversations(user_id)
        
        # 收集最近的对话轮次
        recent_turns = []
        for conv in sorted(conversations, key=lambda x: x.updated_at, reverse=True):
            for turn in conv.turns:
                recent_turns.append({
                    "conversation_id": conv.conversation_id,
                    "question": turn.question,
                    "answer": turn.answer[:500],  # 截断
                    "confidence": turn.confidence,
                    "timestamp": turn.timestamp
                })
                if len(recent_turns) >= max_turns:
                    break
            if len(recent_turns) >= max_turns:
                break
        
        # 查找相似问题
        similar = self.find_similar_questions(current_question, user_id, top_k=3)
        similar_qa = [
            {
                "question": turn.question,
                "answer": turn.answer[:300],
                "confidence": turn.confidence,
                "similarity": score
            }
            for turn, score in similar
            if score > 0.5  # 只返回高相似度的
        ]
        
        return {
            "recent_history": recent_turns,
            "similar_questions": similar_qa,
            "total_conversations": len(conversations),
            "total_turns": sum(len(c.turns) for c in conversations)
        }
    
    def get_all_user_documents_cache(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户所有对话中引用过的文档缓存
        
        Args:
            user_id: 用户ID
            
        Returns:
            文档引用列表
        """
        documents = []
        seen_docs = set()
        
        for conv in self._conversations.values():
            if conv.user_id != user_id:
                continue
            
            for turn in conv.turns:
                for cite in turn.citations:
                    doc_id = cite.get("doc_id", "")
                    if doc_id and doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        documents.append({
                            "doc_id": doc_id,
                            "source": cite.get("source", ""),
                            "first_used": turn.timestamp,
                            "conversation_id": conv.conversation_id
                        })
        
        return documents
    
    # ============================================================
    # 报告历史管理
    # ============================================================
    
    def save_report(self, report: GeneratedReport) -> bool:
        """保存生成的报告"""
        self._reports[report.report_id] = report
        file_path = self.data_dir / f"report_{report.report_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return True
    
    def get_report(self, report_id: str) -> Optional[GeneratedReport]:
        """获取报告"""
        return self._reports.get(report_id)
    
    def get_user_reports(self, user_id: str) -> List[GeneratedReport]:
        """获取用户的所有报告"""
        return [r for r in self._reports.values() if r.user_id == user_id]
    
    def get_conversation_reports(self, conversation_id: str) -> List[GeneratedReport]:
        """获取对话的所有报告"""
        return [r for r in self._reports.values() if r.conversation_id == conversation_id]
    
    def delete_report(self, report_id: str) -> bool:
        """删除报告"""
        if report_id not in self._reports:
            return False
        del self._reports[report_id]
        file_path = self.data_dir / f"report_{report_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True
    
    def get_reflection_context_for_conversation(self, user_id: str, 
                                                  current_question: str) -> Dict[str, Any]:
        """
        获取用户历史反思上下文，用于增强下次对话
        
        Args:
            user_id: 用户ID
            current_question: 当前问题
            
        Returns:
            包含历史反思洞察的上下文
        """
        reflections_context = {
            "key_insights": [],
            "knowledge_gaps": [],
            "user_preferences": {},
            "related_topics": [],
            "follow_up_suggestions": []
        }
        
        # 收集所有对话的反思
        for conv in self._conversations.values():
            if conv.user_id != user_id or not conv.reflection:
                continue
            
            ref = conv.reflection
            
            # 收集关键洞察
            if ref.key_insights:
                reflections_context["key_insights"].extend(ref.key_insights[:3])
            
            # 收集知识空白
            if ref.knowledge_gaps:
                reflections_context["knowledge_gaps"].extend(ref.knowledge_gaps[:2])
            
            # 合并用户偏好
            if ref.user_preferences:
                for key, value in ref.user_preferences.items():
                    if key not in reflections_context["user_preferences"]:
                        reflections_context["user_preferences"][key] = value
            
            # 收集话题
            if ref.topic_summary:
                reflections_context["related_topics"].append(ref.topic_summary[:100])
            
            # 收集后续问题建议
            if ref.follow_up_questions:
                reflections_context["follow_up_suggestions"].extend(ref.follow_up_questions[:2])
        
        # 去重
        reflections_context["key_insights"] = list(set(reflections_context["key_insights"]))[:5]
        reflections_context["knowledge_gaps"] = list(set(reflections_context["knowledge_gaps"]))[:3]
        reflections_context["follow_up_suggestions"] = list(set(reflections_context["follow_up_suggestions"]))[:5]
        
        return reflections_context
