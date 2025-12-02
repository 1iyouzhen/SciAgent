"""
smolagents集成模块 - 将SciAgent与smolagents框架集成
提供：
- smolagents兼容的Tool定义
- CodeAgent集成
- 多Agent协作编排
"""

from typing import Dict, Any, List, Optional
import json


# ============================================================
# smolagents Tool 定义
# ============================================================

try:
    from smolagents import Tool, CodeAgent, ToolCallingAgent
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    # 定义兼容的基类
    class Tool:
        name: str = "base_tool"
        description: str = ""
        inputs: Dict[str, Dict[str, Any]] = {}
        output_type: str = "string"
        
        def forward(self, *args, **kwargs):
            raise NotImplementedError


class DocumentSearchTool(Tool):
    """文档检索工具"""
    
    name = "document_search"
    description = """搜索科学文献数据库，返回与查询相关的文档片段。
    输入查询文本，返回相关的文档内容和来源信息。
    适用于：查找特定主题的文献、获取研究背景、寻找证据支持。"""
    
    inputs = {
        "query": {
            "type": "string",
            "description": "搜索查询文本"
        },
        "top_k": {
            "type": "integer",
            "description": "返回结果数量，默认5",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, vector_db=None):
        super().__init__()
        self.vector_db = vector_db
    
    def forward(self, query: str, top_k: int = 5) -> str:
        if self.vector_db is None:
            return "错误：向量数据库未初始化"
        
        results = self.vector_db.search(query, top_k=top_k)
        
        output_parts = []
        for i, r in enumerate(results):
            output_parts.append(
                f"[结果{i+1}] (来源: {r.source}, 页码: {r.page}, 相关度: {r.score:.2f})\n{r.text[:500]}"
            )
        
        return "\n\n".join(output_parts) if output_parts else "未找到相关文档"


class ImageAnalysisTool(Tool):
    """图像分析工具"""
    
    name = "image_analysis"
    description = """分析科学文献中的图像，包括图表、示意图等。
    输入图像路径，返回图像的详细描述和关键信息。
    适用于：理解图表数据、分析实验结果图、解读示意图。"""
    
    inputs = {
        "image_path": {
            "type": "string",
            "description": "图像文件路径"
        },
        "analysis_type": {
            "type": "string",
            "description": "分析类型：describe(描述)/chart(图表分析)/ocr(文字提取)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, vl_client=None):
        super().__init__()
        self.vl_client = vl_client
    
    def forward(self, image_path: str, analysis_type: str = "describe") -> str:
        if self.vl_client is None:
            return "错误：视觉语言模型未初始化"
        
        import os
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在: {image_path}"
        
        try:
            if analysis_type == "chart":
                result = self.vl_client.analyze_chart(image_path)
                return json.dumps(result, ensure_ascii=False, indent=2)
            elif analysis_type == "ocr":
                return self.vl_client.extract_text_from_image(image_path)
            else:
                return self.vl_client.describe_image(image_path)
        except Exception as e:
            return f"图像分析失败: {e}"


class ReasoningTool(Tool):
    """推理工具"""
    
    name = "reasoning"
    description = """基于提供的证据进行推理，生成带引用的答案。
    输入问题和证据，返回推理结果和引用。
    适用于：综合多个证据得出结论、回答复杂问题、生成研究摘要。"""
    
    inputs = {
        "question": {
            "type": "string",
            "description": "需要回答的问题"
        },
        "evidences": {
            "type": "string",
            "description": "证据文本（多条证据用换行分隔）"
        }
    }
    output_type = "string"
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
    
    def forward(self, question: str, evidences: str) -> str:
        if self.llm_client is None:
            return "错误：LLM客户端未初始化"
        
        from .tools.llm_client import Message
        
        prompt = f"""基于以下证据回答问题，并标注引用。

问题：{question}

证据：
{evidences}

请提供详细答案，每个关键论点标注[来源X]。"""
        
        messages = [Message(role="user", content=prompt)]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            return response.content
        except Exception as e:
            return f"推理失败: {e}"


class VerificationTool(Tool):
    """验证工具"""
    
    name = "verification"
    description = """验证答案的准确性和可靠性。
    输入答案和证据，返回验证结果和置信度。
    适用于：检查答案是否有证据支持、评估答案质量、识别潜在问题。"""
    
    inputs = {
        "answer": {
            "type": "string",
            "description": "需要验证的答案"
        },
        "evidences": {
            "type": "string",
            "description": "支持答案的证据"
        }
    }
    output_type = "string"
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
    
    def forward(self, answer: str, evidences: str) -> str:
        if self.llm_client is None:
            # 使用规则验证
            return self._rule_based_verify(answer, evidences)
        
        from .tools.llm_client import Message
        
        prompt = f"""请验证以下答案是否准确，是否有充分的证据支持。

答案：{answer}

证据：
{evidences}

请输出JSON格式：
{{"confidence": 0.8, "issues": ["问题1"], "suggestions": ["建议1"]}}"""
        
        messages = [Message(role="user", content=prompt)]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.2)
            return response.content
        except Exception as e:
            return self._rule_based_verify(answer, evidences)
    
    def _rule_based_verify(self, answer: str, evidences: str) -> str:
        import re
        
        # 简单规则检查
        citation_count = len(re.findall(r'\[来源\d+\]|\[\d+\]', answer))
        answer_len = len(answer)
        
        confidence = 0.5
        issues = []
        
        if citation_count == 0:
            issues.append("答案缺少引用")
            confidence -= 0.2
        if answer_len < 100:
            issues.append("答案可能不够完整")
            confidence -= 0.1
        
        return json.dumps({
            "confidence": max(0, confidence),
            "issues": issues,
            "suggestions": ["增加引用" if citation_count == 0 else ""]
        }, ensure_ascii=False)


# ============================================================
# smolagents Agent 集成
# ============================================================

class SciAgentCodeAgent:
    """
    基于smolagents CodeAgent的科学文献问答Agent
    
    使用代码生成方式调用工具完成任务
    """
    
    def __init__(self, 
                 vector_db=None,
                 llm_client=None,
                 vl_client=None,
                 model=None):
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.vl_client = vl_client
        
        # 创建工具
        self.tools = [
            DocumentSearchTool(vector_db),
            ImageAnalysisTool(vl_client),
            ReasoningTool(llm_client),
            VerificationTool(llm_client)
        ]
        
        # 创建CodeAgent（如果smolagents可用）
        self.agent = None
        if SMOLAGENTS_AVAILABLE and model:
            self.agent = CodeAgent(tools=self.tools, model=model)
    
    def run(self, question: str) -> Dict[str, Any]:
        """运行Agent"""
        if self.agent:
            # 使用smolagents CodeAgent
            result = self.agent.run(question)
            return {"answer": result, "method": "smolagents"}
        else:
            # 回退到手动工具调用
            return self._manual_run(question)
    
    def _manual_run(self, question: str) -> Dict[str, Any]:
        """手动工具调用流程"""
        # 1. 检索
        search_tool = self.tools[0]
        evidences = search_tool.forward(question, top_k=5)
        
        # 2. 推理
        reasoning_tool = self.tools[2]
        answer = reasoning_tool.forward(question, evidences)
        
        # 3. 验证
        verify_tool = self.tools[3]
        verification = verify_tool.forward(answer, evidences)
        
        return {
            "answer": answer,
            "evidences": evidences,
            "verification": verification,
            "method": "manual"
        }


class MultiAgentOrchestrator:
    """
    多Agent编排器 - 协调多个Agent完成复杂任务
    
    支持：
    - 顺序执行
    - 并行执行
    - 条件分支
    - 迭代循环
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []
    
    def register(self, name: str, agent: Any) -> None:
        """注册Agent"""
        self.agents[name] = agent
    
    def run_sequential(self, 
                       agent_names: List[str], 
                       initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """顺序执行多个Agent"""
        context = initial_input.copy()
        
        for name in agent_names:
            agent = self.agents.get(name)
            if agent is None:
                continue
            
            # 执行Agent
            if hasattr(agent, 'run'):
                result = agent.run(context)
            elif callable(agent):
                result = agent(context)
            else:
                continue
            
            # 更新上下文
            if isinstance(result, dict):
                context.update(result)
            
            # 记录日志
            self.execution_log.append({
                "agent": name,
                "input": str(context)[:200],
                "output": str(result)[:200]
            })
        
        return context
    
    def run_parallel(self,
                     agent_names: List[str],
                     input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并行执行多个Agent"""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
                futures = {}
                for name in agent_names:
                    agent = self.agents.get(name)
                    if agent and hasattr(agent, 'run'):
                        futures[executor.submit(agent.run, input_data)] = name
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        results.append({"agent": name, "result": result})
                    except Exception as e:
                        results.append({"agent": name, "error": str(e)})
            
            return results
        except ImportError:
            # 回退到顺序执行
            return [
                {"agent": name, "result": self.agents[name].run(input_data)}
                for name in agent_names
                if name in self.agents
            ]
    
    def run_iterative(self,
                      agent_name: str,
                      input_data: Dict[str, Any],
                      max_iterations: int = 3,
                      stop_condition: callable = None) -> Dict[str, Any]:
        """迭代执行Agent直到满足条件"""
        agent = self.agents.get(agent_name)
        if agent is None:
            return input_data
        
        context = input_data.copy()
        
        for i in range(max_iterations):
            result = agent.run(context) if hasattr(agent, 'run') else agent(context)
            
            if isinstance(result, dict):
                context.update(result)
            
            # 检查停止条件
            if stop_condition and stop_condition(context):
                break
            
            context["iteration"] = i + 1
        
        return context


# ============================================================
# 便捷函数
# ============================================================

def create_sci_agent(config: Dict[str, Any] = None) -> SciAgentCodeAgent:
    """
    创建SciAgent实例
    
    Args:
        config: 配置字典
        
    Returns:
        SciAgentCodeAgent实例
    """
    from .main import SciAgentPipeline
    
    pipeline = SciAgentPipeline(config)
    
    return SciAgentCodeAgent(
        vector_db=pipeline.vector_db,
        llm_client=pipeline.llm_client,
        vl_client=pipeline.vl_client
    )


def get_available_tools() -> List[Tool]:
    """获取所有可用工具"""
    return [
        DocumentSearchTool(),
        ImageAnalysisTool(),
        ReasoningTool(),
        VerificationTool()
    ]
