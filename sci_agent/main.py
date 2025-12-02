"""
SciAgent - 可验证的科学文献问答系统

核心特性：
- Verifiable（可验证）：强制引用 + 置信度 + 自我校验
- Retrieval-Augmented（检索增强）：Qwen3-Embedding + 多模态混合检索
- Agent-Collaborative（Agent协作）：smolagents多智能体协同
- Iterative：Reviewer → Retriever 迭代优化

工作流程：
用户问题 → Planner → [Retriever, Caption, Reasoner] → Reviewer → 最终输出
"""

import os
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineOutput:
    """流水线输出结果"""
    question: str
    sub_tasks: List[Dict[str, Any]]
    final_answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    iterate_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "sub_tasks": self.sub_tasks,
            "final_answer": self.final_answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "iterate_count": self.iterate_count,
            "metadata": self.metadata
        }
    
    def format_output(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 60,
            "📋 问题",
            "-" * 60,
            self.question,
            "",
            "=" * 60,
            "📝 答案",
            "-" * 60,
            self.final_answer,
            "",
            "=" * 60,
            f"📊 置信度: {self.confidence:.2%}",
            f"🔄 迭代次数: {self.iterate_count}",
            "",
            "=" * 60,
            "📚 引用",
            "-" * 60,
        ]
        
        for i, cite in enumerate(self.citations[:10]):
            source = cite.get("source", "未知来源")
            page = cite.get("page", "?")
            quote = cite.get("quote", "")[:100]
            lines.append(f"[{i+1}] {source} (p.{page})")
            if quote:
                lines.append(f"    \"{quote}...\"")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def _load_config() -> Dict[str, Any]:
    """加载配置文件"""
    defaults = {
        "data": {
            "pdf_dir": "data/pdfs",
            "processed_dir": "data/processed",
            "index_path": "data/index.faiss"
        },
        "models": {
            "embed_model": "Qwen/Qwen3-Embedding-0.6B",
            "vl_model": "glm-4v-flash",
            "reasoner_model": "glm-4-flash",
            "api_provider": "dashscope",
            "api_base": None,  # 让 LLMClient 根据 api_provider 自动选择
            "api_key": ""
        },
        "pipeline": {
            "top_k": 10,
            "max_iterations": 3,
            "confidence_threshold": 0.6,
            "chunk_size": 1024,
            "chunk_overlap": 50
        },
        "agents": {
            "planner": {"max_subtasks": 5},
            "retriever": {"rerank": True, "hybrid": True},
            "reviewer": {"use_llm_judge": True}
        }
    }
    
    try:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                # 深度合并配置
                for k, v in defaults.items():
                    if k not in cfg:
                        cfg[k] = v
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if kk not in cfg[k]:
                                cfg[k][kk] = vv
                return cfg
    except Exception as e:
        print(f"[Warning] 配置加载失败: {e}")
    
    return defaults


def _ensure_dirs(paths: List[str]) -> None:
    """确保目录存在"""
    for d in paths:
        os.makedirs(d, exist_ok=True)


def _resolve_path(path: str, base_dir: Path) -> Path:
    """解析路径"""
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / path


class SciAgentPipeline:
    """
    科学文献问答流水线
    
    实现多智能体协作的完整流程
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or _load_config()
        self.base_dir = Path(__file__).parent
        
        # 初始化路径
        self.pdf_dir = _resolve_path(self.config["data"]["pdf_dir"], self.base_dir)
        self.processed_dir = _resolve_path(self.config["data"]["processed_dir"], self.base_dir)
        self.index_path = _resolve_path(self.config["data"]["index_path"], self.base_dir)
        
        _ensure_dirs([str(self.pdf_dir), str(self.processed_dir)])
        
        # 延迟初始化组件
        self._llm_client = None
        self._vl_client = None
        self._vector_db = None
        self._agents = {}
    
    @property
    def llm_client(self):
        """LLM客户端（延迟加载）"""
        if self._llm_client is None:
            from .tools.llm_client import LLMClient
            self._llm_client = LLMClient(
                model=self.config["models"]["reasoner_model"],
                api_provider=self.config["models"].get("api_provider", "dashscope"),
                api_base=self.config["models"].get("api_base"),
                api_key=self.config["models"].get("api_key")
            )
        return self._llm_client
    
    @property
    def vl_client(self):
        """VL客户端（延迟加载）"""
        if self._vl_client is None:
            from .tools.llm_client import VLClient
            self._vl_client = VLClient(
                model=self.config["models"]["vl_model"],
                api_provider=self.config["models"].get("api_provider", "dashscope"),
                api_base=self.config["models"].get("api_base"),
                api_key=self.config["models"].get("api_key")
            )
        return self._vl_client
    
    @property
    def vector_db(self):
        """向量数据库（延迟加载）"""
        if self._vector_db is None:
            from .tools.vector_db import VectorDB
            self._vector_db = VectorDB(
                embed_model=self.config["models"]["embed_model"],
                index_path=str(self.index_path)
            )
            # 尝试加载已有索引
            loaded = self._vector_db.load_index()
            if not loaded:
                print(f"[Warning] 索引加载失败，请先运行 build_index 构建索引")
            else:
                print(f"[Info] 索引加载成功，文档数: {len(self._vector_db.docs)}")
        return self._vector_db
    
    def _get_agent(self, name: str):
        """获取Agent（延迟加载）"""
        if name not in self._agents:
            from .agents import (
                PlannerAgent, RetrieverAgent, CaptionAgent,
                MultiLLMReasonerAgent, ReviewerAgent, IterativeReviewer
            )
            
            agent_config = self.config.get("agents", {}).get(name, {})
            
            if name == "planner":
                self._agents[name] = PlannerAgent(
                    config=agent_config,
                    llm_client=self.llm_client
                )
            elif name == "retriever":
                self._agents[name] = RetrieverAgent(
                    db=self.vector_db,
                    config={**agent_config, "top_k": self.config["pipeline"]["top_k"]},
                    llm_client=self.llm_client
                )
            elif name == "caption":
                self._agents[name] = CaptionAgent(
                    config=agent_config,
                    vl_client=self.vl_client
                )
            elif name == "reasoner":
                # 使用 MultiLLMReasonerAgent 替代原 ReasonerAgent
                # 从配置中读取多LLM设置
                reasoning_config = self.config.get("agents", {}).get("reasoning", {})
                
                # 创建文本推理LLM
                text_llm = self.llm_client  # 默认使用主LLM
                math_llm = None
                
                # 如果配置了专门的文本推理模型
                if reasoning_config.get("text_reasoner"):
                    text_cfg = reasoning_config["text_reasoner"]
                    from .tools.llm_client import LLMClient
                    text_llm = LLMClient(
                        model=text_cfg.get("model", self.config["models"]["reasoner_model"]),
                        api_provider=text_cfg.get("provider", self.config["models"]["api_provider"])
                    )
                    print(f"[Info] 文本推理模型: {text_cfg.get('model')} ({text_cfg.get('provider')})")
                
                # 如果配置了数学推理模型
                if reasoning_config.get("math_reasoner"):
                    math_cfg = reasoning_config["math_reasoner"]
                    from .tools.llm_client import LLMClient
                    math_llm = LLMClient(
                        model=math_cfg.get("model"),
                        api_provider=math_cfg.get("provider", self.config["models"]["api_provider"])
                    )
                    print(f"[Info] 数学推理模型: {math_cfg.get('model')} ({math_cfg.get('provider')})")
                
                # 打印集成模型配置
                if reasoning_config.get("ensemble_models"):
                    print(f"[Info] 集成推理模型配置:")
                    for em in reasoning_config["ensemble_models"]:
                        print(f"       - {em.get('model')} (权重: {em.get('weight', 0)})")
                
                # 打印调试信息
                print(f"[Debug] reasoning_config.enable_ensemble: {reasoning_config.get('enable_ensemble')}")
                print(f"[Debug] reasoning_config.strategy: {reasoning_config.get('strategy')}")
                print(f"[Debug] reasoning_config.ensemble_models count: {len(reasoning_config.get('ensemble_models', []))}")
                
                self._agents[name] = MultiLLMReasonerAgent(
                    config={"reasoning": reasoning_config},  # 直接传递 reasoning 配置
                    text_llm_client=text_llm,
                    math_llm_client=math_llm,
                    llm_client=text_llm  # 兼容参数名
                )
            elif name == "reviewer":
                self._agents[name] = IterativeReviewer(
                    config={**agent_config, "confidence_threshold": self.config["pipeline"]["confidence_threshold"]},
                    llm_client=self.llm_client,
                    retriever=self._get_agent("retriever")
                )
        
        return self._agents[name]
    
    def build_index(self, pdf_dir: str = None) -> int:
        """
        构建文档索引
        Args:
            pdf_dir: PDF目录路径
            
        Returns:
            索引的文档块数量
        """
        from .tools.pdf_parser import PdfParser
        
        pdf_dir = Path(pdf_dir) if pdf_dir else self.pdf_dir
        parser = PdfParser(output_dir=str(self.processed_dir))
        
        all_chunks = []
        chunk_size = self.config["pipeline"]["chunk_size"]
        chunk_overlap = self.config["pipeline"]["chunk_overlap"]
        
        # 遍历PDF目录
        for file_path in pdf_dir.iterdir():
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            if suffix not in [".pdf", ".txt"]:
                continue
            
            print(f"[Info] 解析文档: {file_path.name}")
            
            try:
                doc = parser.parse(str(file_path))
                chunks = parser.to_chunks(doc, chunk_size=chunk_size, overlap=chunk_overlap)
                all_chunks.extend(chunks)
                print(f"  -> 生成 {len(chunks)} 个文档块")
            except Exception as e:
                print(f"  -> 解析失败: {e}")
        
        # 构建索引
        if all_chunks:
            print(f"[Info] 构建向量索引，共 {len(all_chunks)} 个文档块...")
            self.vector_db.index_documents(all_chunks)
            print("[Info] 索引构建完成")
        
        return len(all_chunks)
    
    def run(self, question: str) -> PipelineOutput:
        """
        运行问答流水线
        
        Args:
            question: 用户问题
            
        Returns:
            PipelineOutput对象
        """
        from .agents.base import AgentContext
        
        # 初始化上下文
        context = AgentContext(
            question=question,
            max_iterations=self.config["pipeline"]["max_iterations"]
        )
        
        print(f"\n{'='*60}")
        print(f"📋 问题: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Planner - 任务分解
        print("[Step 1] Planner: 任务分解...")
        planner = self._get_agent("planner")
        planner_result = planner.run(context)
        if planner_result.success:
            context.sub_tasks = planner_result.data.get("sub_tasks", [])
            print(f"  -> 分解为 {len(context.sub_tasks)} 个子任务")
            for task in context.sub_tasks:
                print(f"     - {task.get('task', '')[:50]}")
        
        # Step 2: Retriever - 检索
        print("\n[Step 2] Retriever: 多模态检索...")
        retriever = self._get_agent("retriever")
        retriever_result = retriever.run(context)
        if retriever_result.success:
            context.evidences = retriever_result.data.get("evidences", [])
            print(f"  -> 检索到 {len(context.evidences)} 条证据")
        
        # Step 3: Caption - 图像理解
        print("\n[Step 3] Caption: 图像理解...")
        # 先统计证据中的图像数量
        image_evidences = [ev for ev in context.evidences if ev.get("chunk_type") == "image"]
        print(f"[Info] VLM配置: provider={self.config['models'].get('api_provider')}, model={self.config['models'].get('vl_model')}")
        
        if len(image_evidences) == 0:
            # 检查是否是因为PDF解析器不支持图像
            has_pypdf_docs = any(
                ev.get("metadata", {}).get("image_support") == False 
                for ev in context.evidences
            )
            if has_pypdf_docs or len(context.evidences) > 0:
                print(f"[Info] 证据中没有图像类型。可能原因：")
                print(f"       1. PDF使用PyPDF2解析，不支持图像提取")
                print(f"       2. 如需图像理解，请安装MinerU: pip install magic-pdf")
                print(f"       3. 安装后需重新构建索引: python -m sci_agent.main --build-index")
        else:
            print(f"[Info] 证据中图像类型数量: {len(image_evidences)}")
        
        caption_agent = self._get_agent("caption")
        caption_result = caption_agent.run(context)
        if caption_result.success:
            context.captions = caption_result.data.get("captions", [])
            print(f"  -> 处理了 {len(context.captions)} 张图像")
        
        # Step 4: Reasoner - 推理生成（使用 MultiLLMReasonerAgent）
        print("\n[Step 4] Reasoner: 推理生成...")
        reasoner = self._get_agent("reasoner")
        reasoner_result = reasoner.run(context)
        if reasoner_result.success:
            data = reasoner_result.data
            context.draft_answer = data.get("answer", "")
            # 从 reasoning_trace 或直接从 data 获取 citations
            if "reasoning_trace" in data and hasattr(data["reasoning_trace"], "document_sources"):
                # 转换 document_sources 为 citations 格式
                trace = data["reasoning_trace"]
                context.citations = []
                for i, src in enumerate(trace.document_sources):
                    context.citations.append({
                        "id": i + 1,
                        "source": src.get("source", ""),
                        "doc_id": src.get("doc_id", ""),
                        "page": src.get("pages", [0])[0] if src.get("pages") else 0
                    })
            else:
                context.citations = data.get("citations", [])
            print(f"  -> 生成答案，包含 {len(context.citations)} 条引用")
        
        # Step 5: Reviewer - 自我校验（带迭代）
        print("\n[Step 5] Reviewer: 自我校验...")
        reviewer = self._get_agent("reviewer")
        
        # 使用迭代审核
        if hasattr(reviewer, 'run_with_iteration'):
            context = reviewer.run_with_iteration(context)
        else:
            reviewer_result = reviewer.run(context)
            if reviewer_result.success:
                context.confidence = reviewer_result.data.get("confidence", 0.0)
                context.draft_answer = reviewer_result.data.get("final_answer", context.draft_answer)
        
        print(f"  -> 置信度: {context.confidence:.2%}")
        print(f"  -> 迭代次数: {context.iteration}")
        
        # 构建输出
        output = PipelineOutput(
            question=question,
            sub_tasks=context.sub_tasks,
            final_answer=context.draft_answer,
            citations=context.citations,
            confidence=context.confidence,
            iterate_count=context.iteration,
            metadata={
                "evidence_count": len(context.evidences),
                "caption_count": len(context.captions)
            }
        )
        
        return output


# 兼容旧接口
def build_index(pdf_dir: str) -> List[Dict[str, Any]]:
    """构建索引（兼容旧接口）"""
    pipeline = SciAgentPipeline()
    count = pipeline.build_index(pdf_dir)
    return [{"count": count}]


def run_pipeline(question: str) -> Dict[str, Any]:
    """运行流水线（兼容旧接口）"""
    pipeline = SciAgentPipeline()
    output = pipeline.run(question)
    return output.to_dict()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SciAgent - 可验证的科学文献问答系统")
    parser.add_argument("--question", "-q", type=str, help="问题")
    parser.add_argument("--build-index", "-b", action="store_true", help="构建索引")
    parser.add_argument("--pdf-dir", type=str, help="PDF目录")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    pipeline = SciAgentPipeline()
    
    # 构建索引
    if args.build_index:
        pdf_dir = args.pdf_dir or str(pipeline.pdf_dir)
        print(f"[Info] 构建索引: {pdf_dir}")
        count = pipeline.build_index(pdf_dir)
        print(f"[Info] 索引完成，共 {count} 个文档块")
        return
    
    # 交互模式
    if args.interactive:
        print("=" * 60)
        print("SciAgent - 可验证的科学文献问答系统")
        print("输入问题进行查询，输入 'quit' 退出")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n📋 请输入问题: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    print("再见！")
                    break
                if not question:
                    continue
                
                output = pipeline.run(question)
                print("\n" + output.format_output())
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"[Error] {e}")
        return
    
    # 单次查询
    question = args.question or os.environ.get("SCI_QUESTION", "请总结文献的关键结论并给出引用。")
    output = pipeline.run(question)
    print("\n" + output.format_output())


if __name__ == "__main__":
    main()
