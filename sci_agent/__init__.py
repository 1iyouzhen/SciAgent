"""
SciAgent - 可验证的科学文献问答系统

核心特性：
- Verifiable（可验证）：强制引用 + 置信度 + 自我校验
- Retrieval-Augmented（检索增强）：Qwen3-Embedding + 多模态混合检索  
- Agent-Collaborative（Agent协作）：smolagents多智能体协同
- Iterative：Reviewer → Retriever 迭代优化

使用示例：
    from sci_agent import SciAgentPipeline
    
    # 创建流水线
    pipeline = SciAgentPipeline()
    
    # 构建索引
    pipeline.build_index("path/to/pdfs")
    
    # 运行问答
    result = pipeline.run("请总结文献的关键结论")
    print(result.format_output())
"""

__version__ = "0.1.0"
__author__ = "SciAgent Team"

from .main import SciAgentPipeline, PipelineOutput, run_pipeline, build_index

__all__ = [
    "SciAgentPipeline",
    "PipelineOutput", 
    "run_pipeline",
    "build_index",
]
