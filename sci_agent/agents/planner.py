"""
Planner Agent - 任务分解
职责：将用户问题分解为可执行的子任务
"""
import re
import json
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentContext, AgentResult


PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是将用户的复杂问题分解为多个可独立执行的子任务。

## 分解原则

1. **细粒度分解**：每个子任务应该聚焦于一个具体的知识点或研究方向
2. **检索友好**：子任务描述应包含明确的关键词，便于文献检索
3. **逻辑递进**：子任务之间应有逻辑顺序（背景→方法→实验→结论）
4. **覆盖全面**：确保子任务能覆盖问题的各个方面

## 分解策略

对于**研究类问题**（如"XX的表现/能力/效果"）：
- 分解为：定义/背景 → 具体方法/技术 → 实验数据/基准测试 → 对比分析 → 局限性/未来方向

对于**比较类问题**（如"A和B的区别"）：
- 分解为：A的特点 → B的特点 → 具体维度1的对比 → 具体维度2的对比 → 综合结论

对于**原理类问题**（如"为什么/如何"）：
- 分解为：基本概念 → 核心机制 → 关键组件 → 工作流程 → 应用场景

## 子任务数量

- 简单问题：2-3个子任务
- 中等问题：4-6个子任务  
- 复杂研究问题：6-8个子任务

## 输出格式（JSON）

{
    "analysis": "问题分析：识别问题类型、核心概念、需要覆盖的方面",
    "complexity": "simple/medium/complex",
    "sub_tasks": [
        {
            "id": 1, 
            "task": "具体、可检索的子任务描述", 
            "type": "retrieval/reasoning/comparison/summary",
            "keywords": ["关键词1", "关键词2"],
            "focus": "该子任务聚焦的具体方面"
        }
    ]
}"""


class PlannerAgent(BaseAgent):
    """
    Planner Agent - 任务分解
    
    功能：
    - 分析用户问题
    - 分解为可执行子任务
    - 确定任务类型和执行顺序
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_client = None):
        super().__init__(name="planner", config=config)
        self.llm_client = llm_client
        self.max_subtasks = config.get("max_subtasks", 5) if config else 5
    
    def run(self, context: AgentContext) -> AgentResult:
        """
        执行任务分解
        
        Args:
            context: Agent上下文
            
        Returns:
            包含子任务列表的结果
        """
        question = context.question
        
        if not question.strip():
            print(f"  [Planner] 错误: 问题为空")
            return AgentResult(success=False, error="问题为空")
        
        print(f"  [Planner] 开始分解问题: {question[:50]}...")
        
        # 尝试使用LLM进行智能分解
        if self.llm_client:
            print(f"  [Planner] 使用LLM进行智能分解")
            sub_tasks = self._plan_with_llm(question)
        else:
            # 回退到规则分解
            print(f"  [Planner] 使用规则进行分解")
            sub_tasks = self._plan_with_rules(question)
        
        print(f"  [Planner] 分解完成，生成 {len(sub_tasks)} 个子任务")
        for i, task in enumerate(sub_tasks[:3]):
            print(f"    - 子任务{i+1}: {task.get('task', '')[:40]}...")
        
        return AgentResult(
            success=True,
            data={"sub_tasks": sub_tasks}
        )
    
    def _plan_with_llm(self, question: str) -> List[Dict[str, Any]]:
        """使用LLM进行任务分解"""
        from ..tools.llm_client import Message
        
        messages = [
            Message(role="system", content=PLANNER_SYSTEM_PROMPT),
            Message(role="user", content=f"请分解以下问题：\n\n{question}")
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            
            # 解析JSON响应
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                tasks = result.get("sub_tasks", [])
                complexity = result.get("complexity", "medium")
                
                # 根据复杂度调整最大子任务数
                max_tasks = {"simple": 3, "medium": 6, "complex": 8}.get(complexity, self.max_subtasks)
                
                # 标准化格式，生成多个检索查询
                standardized_tasks = []
                for i, t in enumerate(tasks[:max_tasks]):
                    task_desc = t.get("task", "")
                    keywords = t.get("keywords", [])
                    focus = t.get("focus", "")
                    
                    # 构建更丰富的检索查询
                    query_parts = [task_desc]
                    if keywords:
                        query_parts.append(" ".join(keywords))
                    if focus:
                        query_parts.append(focus)
                    
                    standardized_tasks.append({
                        "id": t.get("id", i+1),
                        "task": task_desc,
                        "type": t.get("type", "retrieval"),
                        "query": " ".join(query_parts),  # 更丰富的检索查询
                        "keywords": keywords,
                        "focus": focus
                    })
                
                return standardized_tasks
        except Exception as e:
            print(f"[Warning] LLM任务分解失败: {e}")
        
        # 回退到规则分解
        return self._plan_with_rules(question)
    
    def _plan_with_rules(self, question: str) -> List[Dict[str, Any]]:
        """基于规则的任务分解"""
        sub_tasks = []
        
        # 1. 按标点符号分割
        parts = self._split_by_punctuation(question)
        
        # 2. 识别任务类型
        for i, part in enumerate(parts[:self.max_subtasks]):
            task_type = self._identify_task_type(part)
            sub_tasks.append({
                "id": i + 1,
                "task": part,
                "type": task_type,
                "query": self._extract_query(part)
            })
        
        # 3. 如果没有分割出子任务，使用原问题
        if not sub_tasks:
            sub_tasks.append({
                "id": 1,
                "task": question,
                "type": "retrieval",
                "query": question
            })
        
        return sub_tasks
    
    def _split_by_punctuation(self, text: str) -> List[str]:
        """按标点符号分割文本"""
        # 分割符
        separators = r'[。？！；\n]|(?<=[.?!;])\s+'
        parts = re.split(separators, text)
        
        # 清理并过滤空白
        result = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 2:  # 过滤太短的片段
                result.append(part)
        
        return result
    
    def _identify_task_type(self, task: str) -> str:
        """识别任务类型"""
        task_lower = task.lower()
        
        # 比较类任务
        comparison_keywords = ["比较", "对比", "区别", "不同", "相同", "差异", "versus", "vs", "compare"]
        if any(kw in task_lower for kw in comparison_keywords):
            return "comparison"
        
        # 推理类任务
        reasoning_keywords = ["为什么", "原因", "解释", "分析", "推断", "证明", "why", "explain", "analyze"]
        if any(kw in task_lower for kw in reasoning_keywords):
            return "reasoning"
        
        # 总结类任务
        summary_keywords = ["总结", "概括", "摘要", "归纳", "summarize", "summary"]
        if any(kw in task_lower for kw in summary_keywords):
            return "summary"
        
        # 默认为检索类
        return "retrieval"
    
    def _extract_query(self, task: str) -> str:
        """提取检索查询"""
        # 移除疑问词和修饰词
        query = task
        
        # 移除常见的疑问词前缀
        prefixes = ["请问", "请", "能否", "可以", "帮我", "告诉我"]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):]
        
        return query.strip()
    
    # 兼容旧接口
    def plan(self, question: str) -> List[str]:
        """
        兼容旧接口的任务分解方法
        
        Args:
            question: 用户问题
            
        Returns:
            子任务字符串列表
        """
        context = AgentContext(question=question)
        result = self.run(context)
        
        if result.success:
            return [t["task"] for t in result.data.get("sub_tasks", [])]
        return [question]
