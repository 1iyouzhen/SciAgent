"""
Agent基类 - 基于smolagents的多智能体协作框架
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Agent上下文，用于Agent间通信"""
    question: str = ""
    sub_tasks: List[str] = field(default_factory=list)
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    captions: List[Dict[str, Any]] = field(default_factory=list)
    draft_answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    iteration: int = 0
    max_iterations: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "sub_tasks": self.sub_tasks,
            "evidences": self.evidences,
            "captions": self.captions,
            "draft_answer": self.draft_answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "iteration": self.iteration,
            "metadata": self.metadata
        }


@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    
    
class BaseAgent(ABC):
    """
    Agent基类
    
    所有Agent继承此类，实现run方法
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        """
        执行Agent任务
        
        Args:
            context: Agent上下文
            
        Returns:
            执行结果
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class Tool:
    """
    工具基类 - 兼容smolagents Tool接口
    """
    
    name: str = "base_tool"
    description: str = "Base tool"
    inputs: Dict[str, Dict[str, Any]] = {}
    output_type: str = "string"
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class AgentOrchestrator:
    """
    Agent编排器 - 管理多Agent协作
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """注册Agent"""
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取Agent"""
        return self.agents.get(name)
    
    def run_pipeline(self, 
                     question: str,
                     pipeline: List[str] = None) -> AgentContext:
        """
        运行Agent流水线
        
        Args:
            question: 用户问题
            pipeline: Agent执行顺序
            
        Returns:
            最终上下文
        """
        if pipeline is None:
            pipeline = ["planner", "retriever", "caption", "reasoner", "reviewer"]
        
        context = AgentContext(
            question=question,
            max_iterations=self.config.get("max_iterations", 3)
        )
        
        for agent_name in pipeline:
            agent = self.agents.get(agent_name)
            if agent is None:
                print(f"[Warning] Agent not found: {agent_name}")
                continue
            
            result = agent.run(context)
            if not result.success:
                print(f"[Error] Agent {agent_name} failed: {result.error}")
                break
            
            # 更新上下文
            self._update_context(context, agent_name, result)
        
        return context
    
    def _update_context(self, 
                        context: AgentContext, 
                        agent_name: str, 
                        result: AgentResult) -> None:
        """更新上下文"""
        data = result.data
        
        if agent_name == "planner":
            context.sub_tasks = data.get("sub_tasks", [])
        elif agent_name == "retriever":
            context.evidences = data.get("evidences", [])
        elif agent_name == "caption":
            context.captions = data.get("captions", [])
        elif agent_name == "reasoner":
            context.draft_answer = data.get("answer", "")
            context.citations = data.get("citations", [])
        elif agent_name == "reviewer":
            context.confidence = data.get("confidence", 0.0)
            if data.get("final_answer"):
                context.draft_answer = data["final_answer"]


def create_smolagent_tool(func, name: str, description: str, inputs: Dict) -> Tool:
    """
    创建smolagents兼容的工具
    
    Args:
        func: 工具函数
        name: 工具名称
        description: 工具描述
        inputs: 输入参数定义
        
    Returns:
        Tool对象
    """
    class DynamicTool(Tool):
        pass
    
    DynamicTool.name = name
    DynamicTool.description = description
    DynamicTool.inputs = inputs
    DynamicTool.forward = staticmethod(func)
    
    return DynamicTool()
