"""
Agents模块 - 多智能体协作系统

包含以下Agent：
- PlannerAgent: 任务分解
- RetrieverAgent: 多模态混合检索
- CaptionAgent: 图像理解
- MultiLLMReasonerAgent: 多LLM协同推理（整合了原ReasonerAgent）
- ReviewerAgent: 自我校验
- DeepThinkerAgent: 深度思考（类似DeepSeek-R1）
- ReflectorAgent: 自我反思
- ReportGeneratorAgent: 综述报告生成
- ImageGeneratorAgent: 图片生成与理解

注意：ReasonerAgent 和 ChainOfThoughtReasoner 现在是 MultiLLMReasonerAgent 的别名
"""

from .base import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentOrchestrator,
    Tool,
    create_smolagent_tool
)

from .planner import PlannerAgent
from .retriever import RetrieverAgent, MultiModalRetriever
from .caption import CaptionAgent, BatchCaptionAgent
# MultiLLMReasonerAgent 整合了原 ReasonerAgent 的功能
from .multi_llm_reasoner import (
    MultiLLMReasonerAgent,
    ReasonerAgent,  # 兼容别名
    ChainOfThoughtReasoner  # 兼容别名
)
from .reviewer import ReviewerAgent, ReviewResult, IterativeReviewer
from .deep_thinker import DeepThinkerAgent
from .reflector import ReflectorAgent
from .report_generator import ReportGeneratorAgent
from .image_generator import ImageGeneratorAgent


__all__ = [
    # Base
    "BaseAgent",
    "AgentContext", 
    "AgentResult",
    "AgentOrchestrator",
    "Tool",
    "create_smolagent_tool",
    
    # Agents
    "PlannerAgent",
    "RetrieverAgent",
    "MultiModalRetriever",
    "CaptionAgent",
    "BatchCaptionAgent",
    "MultiLLMReasonerAgent",
    "ReasonerAgent",  # 兼容别名 -> MultiLLMReasonerAgent
    "ChainOfThoughtReasoner",  # 兼容别名 -> MultiLLMReasonerAgent
    "ReviewerAgent",
    "ReviewResult",
    "IterativeReviewer",
    "DeepThinkerAgent",
    "ReflectorAgent",
    "ReportGeneratorAgent",
    "ImageGeneratorAgent",
]
