"""
FastAPI后端 - 提供RESTful API接口
"""
import os
import re
import secrets
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 加载 .env 环境变量（override=True 确保覆盖已存在的环境变量）
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[Info] 已加载环境变量: {env_path}")

from fastapi import FastAPI, HTTPException, Depends, Header, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json

# 导入核心模块
from ..main import SciAgentPipeline
from ..models.user import UserManager
from ..models.memory import (
    MemoryStore, Conversation, ConversationTurn, 
    ReasoningTrace, ThinkingStep, Reflection
)
from ..models.document import DocumentManager, DocumentSession, UploadedDocument
from ..agents import (
    DeepThinkerAgent, ReflectorAgent, ReportGeneratorAgent,
    MultiLLMReasonerAgent, ImageGeneratorAgent
)
from ..models.memory import GeneratedReport
from ..agents.base import AgentContext
from ..tools.pdf_parser import PdfParser


# ============================================================
# Pydantic Models
# ============================================================

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    show_thinking: bool = True  # 是否展示推理过程


class RegenerateRequest(BaseModel):
    conversation_id: str
    turn_id: str


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class UrlUploadRequest(BaseModel):
    urls: List[str] = Field(..., max_items=50)
    conversation_id: Optional[str] = None


class ReportRequest(BaseModel):
    conversation_id: str
    requirement: str = Field(..., min_length=1)  # 用户的报告需求


class MindmapRequest(BaseModel):
    conversation_id: str
    doc_ids: List[str] = Field(default_factory=list)  # 选择的文档ID，空则使用全部


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)  # 要翻译的文本
    target_language: str = Field(default="zh")  # 目标语言
    source_language: str = Field(default="auto")  # 源语言


class DocumentTranslateRequest(BaseModel):
    conversation_id: str
    doc_id: str  # 文档ID
    target_language: str = Field(default="zh")  # 目标语言
    start_page: Optional[int] = None  # 起始页码（可选）
    end_page: Optional[int] = None  # 结束页码（可选）
    preserve_latex: bool = True  # 是否保留LaTeX格式


class InterestingContentRequest(BaseModel):
    conversation_id: str


class ImageVisualizationRequest(BaseModel):
    conversation_id: str
    doc_id: str  # 选择的文档ID
    prompt: str = Field(..., min_length=1)  # 用户提示词


class ReportHistoryResponse(BaseModel):
    report_id: str
    requirement: str
    title: str
    confidence: float
    created_at: str


class TokenResponse(BaseModel):
    token: str
    user_id: str
    username: str


class AnswerResponse(BaseModel):
    turn_id: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    reasoning_trace: Optional[Dict[str, Any]] = None
    reflection_available: bool = False


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    turns: List[Dict[str, Any]]
    reflection: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


# ============================================================
# 全局实例
# ============================================================

user_manager: UserManager = None
memory_store: MemoryStore = None
document_manager: DocumentManager = None
pipeline: SciAgentPipeline = None
deep_thinker: DeepThinkerAgent = None
reflector: ReflectorAgent = None
report_generator: ReportGeneratorAgent = None
multi_llm_reasoner: MultiLLMReasonerAgent = None
image_generator: ImageGeneratorAgent = None
pdf_parser: PdfParser = None

# 后台任务线程池
executor: ThreadPoolExecutor = None
# 处理状态跟踪
processing_tasks: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global user_manager, memory_store, document_manager, pipeline
    global deep_thinker, reflector, report_generator, pdf_parser, executor
    global multi_llm_reasoner, image_generator
    
    # 初始化
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    user_manager = UserManager(os.path.join(data_dir, "users"))
    memory_store = MemoryStore(os.path.join(data_dir, "memory"))
    document_manager = DocumentManager(os.path.join(data_dir, "documents"))
    pipeline = SciAgentPipeline()
    pdf_parser = PdfParser(os.path.join(data_dir, "processed"))
    
    # 初始化线程池（用于并行处理文档）
    executor = ThreadPoolExecutor(max_workers=4)
    
    # 初始化Agents
    deep_thinker = DeepThinkerAgent(llm_client=pipeline.llm_client)
    reflector = ReflectorAgent(llm_client=pipeline.llm_client)
    report_generator = ReportGeneratorAgent(llm_client=pipeline.llm_client)
    
    # 初始化多LLM推理Agent - 从配置读取多个模型
    from ..tools.llm_client import LLMClient
    
    reasoning_config = pipeline.config.get("agents", {}).get("reasoning", {})
    
    # 创建文本推理LLM客户端
    text_llm = pipeline.llm_client
    
    # 创建数学推理LLM客户端（如果配置了不同的模型）
    math_config = reasoning_config.get("math_reasoner", {})
    if math_config and math_config.get("model"):
        try:
            math_llm = LLMClient(
                model=math_config.get("model"),
                api_provider=math_config.get("provider", pipeline.config["models"].get("api_provider")),
                api_key=pipeline.config["models"].get("api_key")
            )
            print(f"[Info] 数学推理模型: {math_config.get('model')}")
        except Exception as e:
            print(f"[Warning] 数学推理模型初始化失败: {e}，使用默认模型")
            math_llm = text_llm
    else:
        math_llm = text_llm
    
    # 打印多LLM配置信息
    print(f"[Info] 多LLM推理配置:")
    print(f"  -> 文本推理: {pipeline.config['models'].get('reasoner_model')}")
    print(f"  -> 数学推理: {math_config.get('model', '同上')}")
    print(f"  -> 集成模式: {reasoning_config.get('strategy', {}).get('mode', 'single')}")
    
    multi_llm_reasoner = MultiLLMReasonerAgent(
        config=reasoning_config,
        text_llm_client=text_llm,
        math_llm_client=math_llm,
        vision_llm_client=pipeline.vl_client if hasattr(pipeline, 'vl_client') else None
    )
    
    # 初始化图片生成Agent（配置从环境变量读取）
    # 打印环境变量调试信息
    print(f"[Debug] IMAGE_MODEL 环境变量: {os.environ.get('IMAGE_MODEL', '未设置')}")
    
    image_generator = ImageGeneratorAgent(
        config={
            "output_dir": os.path.join(data_dir, "generated_images")
            # 其他配置从 .env 环境变量读取:
            # IMAGE_API_PROVIDER, IMAGE_MODEL, IMAGE_SIZE, IMAGE_STEPS, IMAGE_GUIDANCE
        },
        llm_client=pipeline.llm_client,
        vl_client=None  # 可配置VL模型
    )
    
    # 预加载 Embedding 模型（后台线程，不阻塞启动）
    def preload_embedding():
        try:
            pipeline.vector_db.encoder.preload()
        except Exception as e:
            print(f"[Warning] 预加载 Embedding 模型失败: {e}")
    
    executor.submit(preload_embedding)
    
    print("[Info] SciAgent API 启动完成")
    yield
    
    # 清理
    if executor:
        executor.shutdown(wait=False)
    print("[Info] SciAgent API 关闭")


app = FastAPI(
    title="SciAgent API",
    description="可验证的科学文献问答系统API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 认证依赖
# ============================================================

async def get_current_user(authorization: str = Header(None)):
    """获取当前用户"""
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    
    # 解析Bearer token
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    user = user_manager.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="无效或过期的令牌")
    
    return user


# ============================================================
# 用户认证API
# ============================================================

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(data: UserRegister):
    """用户注册"""
    user = user_manager.register(data.username, data.password)
    if not user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 自动登录
    session = user_manager.login(data.username, data.password)
    return TokenResponse(
        token=session.session_id,
        user_id=user.user_id,
        username=user.username
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """用户登录"""
    session = user_manager.login(data.username, data.password)
    if not session:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    user = user_manager.get_user(session.user_id)
    return TokenResponse(
        token=session.session_id,
        user_id=user.user_id,
        username=user.username
    )


@app.post("/api/auth/logout")
async def logout(user=Depends(get_current_user), authorization: str = Header(None)):
    """用户登出"""
    token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    user_manager.logout(token)
    return {"message": "登出成功"}


@app.get("/api/auth/me")
async def get_me(user=Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "user_id": user.user_id,
        "username": user.username,
        "created_at": user.created_at
    }


# ============================================================
# 对话管理API
# ============================================================

@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(data: ConversationCreate, user=Depends(get_current_user)):
    """创建新对话"""
    conv = memory_store.create_conversation(user.user_id, data.title or "")
    return ConversationResponse(
        conversation_id=conv.conversation_id,
        title=conv.title,
        turns=[],
        reflection=None,
        created_at=conv.created_at,
        updated_at=conv.updated_at
    )


@app.get("/api/conversations")
async def list_conversations(user=Depends(get_current_user)):
    """获取用户的所有对话"""
    conversations = memory_store.get_user_conversations(user.user_id)
    return [
        {
            "conversation_id": c.conversation_id,
            "title": c.title,
            "turn_count": len(c.turns),
            "created_at": c.created_at,
            "updated_at": c.updated_at
        }
        for c in sorted(conversations, key=lambda x: x.updated_at, reverse=True)
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, user=Depends(get_current_user)):
    """获取对话详情"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    return ConversationResponse(
        conversation_id=conv.conversation_id,
        title=conv.title,
        turns=[t.to_dict() for t in conv.turns],
        reflection=conv.reflection.to_dict() if conv.reflection else None,
        created_at=conv.created_at,
        updated_at=conv.updated_at
    )


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user=Depends(get_current_user)):
    """删除对话"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    memory_store.delete_conversation(conversation_id)
    return {"message": "删除成功"}


# ============================================================
# 问答API
# ============================================================

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest, user=Depends(get_current_user)):
    """提问并获取回答"""
    # 获取或创建对话
    if data.conversation_id:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conv = memory_store.create_conversation(user.user_id)
    
    # 获取用户历史上下文（跨对话记忆）
    history_context = memory_store.get_user_history_context(
        user.user_id, data.question
    )
    
    # 获取反思上下文（用于增强回答）
    reflection_context = memory_store.get_reflection_context_for_conversation(
        user.user_id, data.question
    )
    
    # 检查相似问题（记忆库）- 如果有高相似度的历史回答，可以参考
    similar_questions = history_context.get("similar_questions", [])
    
    # 初始化上下文
    context = AgentContext(question=data.question)
    
    print(f"\n{'='*60}")
    print(f"[Pipeline] 开始处理问题: {data.question[:50]}...")
    print(f"{'='*60}")
    
    # Step 1: Planner - 任务分解
    print(f"\n[Step 1] Planner: 任务分解...")
    planner = pipeline._get_agent("planner")
    planner_result = planner.run(context)
    if planner_result.success:
        context.sub_tasks = planner_result.data.get("sub_tasks", [])
        print(f"  -> 分解为 {len(context.sub_tasks)} 个子任务")
    
    # Step 2: Retriever - 检索证据
    print(f"\n[Step 2] Retriever: 多模态检索...")
    retriever = pipeline._get_agent("retriever")
    retriever_result = retriever.run(context)
    evidences = []
    if retriever_result.success:
        evidences = retriever_result.data.get("evidences", [])
        context.evidences = evidences
        print(f"  -> 检索到 {len(evidences)} 条证据")
    
    # Step 3: Caption - 图像理解（如果有图像证据）
    print(f"\n[Step 3] Caption: 图像理解...")
    caption_agent = pipeline._get_agent("caption")
    caption_result = caption_agent.run(context)
    captions = []
    if caption_result.success:
        captions = caption_result.data.get("captions", [])
        context.captions = captions
        print(f"  -> 处理了 {len(captions)} 张图像")
    
    # Step 4: 推理生成 - 使用多LLM协同推理
    print(f"\n[Step 4] MultiLLMReasoner: 多LLM协同推理...")
    reasoning_trace = None
    final_answer = ""
    confidence = 0.0
    citations = []
    
    # 优先使用 MultiLLMReasoner 进行集成推理
    print(f"  -> 使用多LLM集成推理模式")
    reasoner_result = multi_llm_reasoner.run(context)
    if reasoner_result.success:
        reasoner_data = reasoner_result.data
        final_answer = reasoner_data.get("answer", "")
        confidence = reasoner_data.get("confidence", 0.5)
        analysis_type = reasoner_data.get("analysis_type", "text")
        print(f"  -> 推理类型: {analysis_type}, 置信度: {confidence:.2%}")
        
        trace = reasoner_data.get("reasoning_trace")
        if trace:
            reasoning_trace = trace.to_dict() if hasattr(trace, "to_dict") else trace
            # 从 reasoning_trace 提取 citations
            if isinstance(reasoning_trace, dict):
                doc_sources = reasoning_trace.get("document_sources", [])
                for i, src in enumerate(doc_sources):
                    doc_id = src.get("doc_id") or ""
                    citations.append({
                        "id": i + 1,
                        "source": src.get("source", ""),
                        "doc_id": doc_id,
                        "page": src.get("pages", [0])[0] if src.get("pages") else 0
                    })
    
    # 如果 MultiLLMReasoner 没有生成有效答案，回退到 DeepThinker
    if not final_answer and data.show_thinking:
        print(f"  -> 回退到 DeepThinker 进行深度推理")
        thinker_result = deep_thinker.run(context)
        if thinker_result.success:
            thinker_data = thinker_result.data
            final_answer = thinker_data.get("answer", "")
            confidence = thinker_data.get("confidence", 0.5)
            trace = thinker_data.get("reasoning_trace")
            if trace:
                reasoning_trace = trace.to_dict() if hasattr(trace, "to_dict") else trace
                if hasattr(trace, "document_sources"):
                    for i, src in enumerate(trace.document_sources):
                        doc_id = src.get("doc_id") or ""
                        citations.append({
                            "id": i + 1,
                            "source": src.get("source", ""),
                            "doc_id": doc_id,
                            "page": src.get("pages", [0])[0] if src.get("pages") else 0
                        })
    
    # 如果推理失败，使用简单的证据合成作为回退
    if not final_answer and evidences:
        answer_parts = [f"根据检索到的证据，关于「{data.question}」的相关信息如下：\n"]
        for i, ev in enumerate(evidences[:5]):
            source = ev.get("source", "未知")
            page = ev.get("page", "?")
            text = ev.get("text", "")[:300]
            # 确保 doc_id 不为空
            doc_id = ev.get("doc_id") or ""
            answer_parts.append(f"[来源{i+1}: {source}, p.{page}] {text}")
            citations.append({
                "id": i + 1,
                "source": source,
                "doc_id": doc_id,
                "page": page
            })
        final_answer = "\n\n".join(answer_parts)
        confidence = 0.4
    
    # Step 5: Reviewer - 自我校验（可选）
    if final_answer:
        context.draft_answer = final_answer
        context.citations = citations
        reviewer = pipeline._get_agent("reviewer")
        if hasattr(reviewer, 'run_with_iteration'):
            context = reviewer.run_with_iteration(context)
            confidence = context.confidence
            final_answer = context.draft_answer
        else:
            reviewer_result = reviewer.run(context)
            if reviewer_result.success:
                confidence = reviewer_result.data.get("confidence", confidence)
                final_answer = reviewer_result.data.get("final_answer", final_answer)
    
    # 创建对话轮次
    turn_id = secrets.token_hex(16)
    turn = ConversationTurn(
        turn_id=turn_id,
        question=data.question,
        answer=final_answer,
        reasoning_trace=reasoning_trace if data.show_thinking else None,
        confidence=confidence,
        citations=citations
    )
    
    # 保存到记忆库
    memory_store.add_turn(conv.conversation_id, turn)
    
    # 检查是否可以生成反思（至少1轮对话）
    reflection_available = len(conv.turns) >= 1
    
    return AnswerResponse(
        turn_id=turn_id,
        answer=final_answer,
        confidence=confidence,
        citations=citations,
        reasoning_trace=reasoning_trace,
        reflection_available=reflection_available
    )


@app.post("/api/ask/stream")
async def ask_question_stream(data: QuestionRequest, user=Depends(get_current_user)):
    """流式问答（展示思考过程）"""
    # 获取或创建对话
    if data.conversation_id:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conv = memory_store.create_conversation(user.user_id)
    
    async def generate():
        # 先进行检索
        context = AgentContext(question=data.question)
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 开始处理问题: {data.question[:50]}...")
        print(f"{'='*60}")
        
        yield f"data: {json.dumps({'type': 'status', 'message': '正在分解任务...'})}\n\n"
        
        # Step 1: Planner - 任务分解
        print(f"\n[Step 1] Planner: 任务分解...")
        planner = pipeline._get_agent("planner")
        planner_result = planner.run(context)
        if planner_result.success:
            context.sub_tasks = planner_result.data.get("sub_tasks", [])
            print(f"  -> 分解为 {len(context.sub_tasks)} 个子任务")
            for task in context.sub_tasks[:3]:
                print(f"     - {task.get('task', task.get('query', ''))[:50]}")
        
        yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关文档...'})}\n\n"
        
        # Step 2: Retriever - 检索
        print(f"\n[Step 2] Retriever: 多模态检索...")
        retriever = pipeline._get_agent("retriever")
        retriever_result = retriever.run(context)
        evidences = []
        if retriever_result.success:
            evidences = retriever_result.data.get("evidences", [])
            context.evidences = evidences
            print(f"  -> 检索到 {len(evidences)} 条证据")
            # 显示前3条证据的来源
            for ev in evidences[:3]:
                print(f"     - {ev.get('source', '未知')[:30]} (p.{ev.get('page', '?')}, score={ev.get('score', 0):.2f})")
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'检索到 {len(evidences)} 条相关证据'})}\n\n"
        
        # 用于保存流式推理的最终结果
        final_answer = ""
        confidence = 0.5
        citations = []
        reasoning_trace = None
        
        # Step 3: 推理生成
        print(f"\n[Step 3] Reasoner: 多LLM协同推理...")
        
        # 流式输出思考过程，并捕获最终结果
        if data.show_thinking:
            # 使用多LLM协同推理的流式输出
            print(f"  -> 使用 MultiLLMReasoner 进行集成推理")
            for step in multi_llm_reasoner.reason_stream(data.question, evidences):
                yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
                
                # 捕获最终结果
                if step.get("type") == "final_result":
                    final_answer = step.get("answer", "")
                    confidence = step.get("confidence", 0.5)
                    reasoning_trace = step.get("reasoning_trace")
                    analysis_type = step.get("analysis_type", "text")
                    print(f"  -> 推理类型: {analysis_type}, 置信度: {confidence:.2%}")
                    
                    # 从 reasoning_trace 提取 citations
                    if reasoning_trace and isinstance(reasoning_trace, dict):
                        doc_sources = reasoning_trace.get("document_sources", [])
                        for i, src in enumerate(doc_sources):
                            # 确保 doc_id 不为空
                            doc_id = src.get("doc_id") or ""
                            citations.append({
                                "id": i + 1,
                                "source": src.get("source", ""),
                                "doc_id": doc_id,
                                "page": src.get("pages", [0])[0] if src.get("pages") else 0
                            })
        
        # 如果流式推理没有生成有效答案，使用 DeepThinker 作为回退
        if not final_answer:
            yield f"data: {json.dumps({'type': 'status', 'message': '正在使用深度思考生成答案...'})}\n\n"
            print(f"  -> 回退到 DeepThinker 进行深度推理")
            
            thinker_result = deep_thinker.run(context)
            if thinker_result.success:
                thinker_data = thinker_result.data
                final_answer = thinker_data.get("answer", "")
                confidence = thinker_data.get("confidence", 0.5)
                trace = thinker_data.get("reasoning_trace")
                if trace:
                    reasoning_trace = trace.to_dict() if hasattr(trace, "to_dict") else trace
                    # 从 reasoning_trace 提取 citations
                    if isinstance(reasoning_trace, dict):
                        doc_sources = reasoning_trace.get("document_sources", [])
                        for i, src in enumerate(doc_sources):
                            doc_id = src.get("doc_id") or ""
                            citations.append({
                                "id": i + 1,
                                "source": src.get("source", ""),
                                "doc_id": doc_id,
                                "page": src.get("pages", [0])[0] if src.get("pages") else 0
                            })
        
        # 如果仍然没有答案，使用简单的证据合成
        if not final_answer and evidences:
            answer_parts = [f"根据检索到的证据，关于「{data.question}」的相关信息如下：\n"]
            for i, ev in enumerate(evidences[:5]):
                source = ev.get("source", "未知")
                page = ev.get("page", "?")
                text = ev.get("text", "")[:300]
                # 确保 doc_id 不为空
                doc_id = ev.get("doc_id") or ""
                answer_parts.append(f"[来源{i+1}: {source}, p.{page}] {text}")
                citations.append({
                    "id": i + 1,
                    "source": source,
                    "doc_id": doc_id,
                    "page": page
                })
            final_answer = "\n\n".join(answer_parts)
            confidence = 0.4
        
        # 创建对话轮次
        turn_id = secrets.token_hex(16)
        turn = ConversationTurn(
            turn_id=turn_id,
            question=data.question,
            answer=final_answer,
            reasoning_trace=reasoning_trace,
            confidence=confidence,
            citations=citations
        )
        memory_store.add_turn(conv.conversation_id, turn)
        
        # 发送最终结果
        yield f"data: {json.dumps({'type': 'complete', 'turn_id': turn_id, 'answer': final_answer, 'confidence': confidence, 'citations': citations}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/regenerate", response_model=AnswerResponse)
async def regenerate_answer(data: RegenerateRequest, user=Depends(get_current_user)):
    """重新生成回答"""
    conv = memory_store.get_conversation(data.conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 找到对应的轮次
    target_turn = None
    for turn in conv.turns:
        if turn.turn_id == data.turn_id:
            target_turn = turn
            break
    
    if not target_turn:
        raise HTTPException(status_code=404, detail="对话轮次不存在")
    
    # 重新运行流水线
    output = pipeline.run(target_turn.question)
    
    # 生成新的推理追踪
    context = AgentContext(
        question=target_turn.question,
        evidences=[]
    )
    retriever = pipeline._get_agent("retriever")
    retriever_result = retriever.run(context)
    if retriever_result.success:
        context.evidences = retriever_result.data.get("evidences", [])
    
    thinker_result = deep_thinker.run(context)
    new_trace = thinker_result.data.get("reasoning_trace") if thinker_result.success else None
    
    # 更新答案（保存旧答案到历史）
    memory_store.update_turn_answer(
        data.conversation_id,
        data.turn_id,
        output.final_answer,
        new_trace,
        output.confidence
    )
    
    return AnswerResponse(
        turn_id=data.turn_id,
        answer=output.final_answer,
        confidence=output.confidence,
        citations=output.citations,
        reasoning_trace=new_trace.to_dict() if new_trace else None,
        reflection_available=True
    )


@app.get("/api/conversations/{conversation_id}/turns/{turn_id}/history")
async def get_answer_history(conversation_id: str, turn_id: str, user=Depends(get_current_user)):
    """获取回答历史（包括重新生成的版本）"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    for turn in conv.turns:
        if turn.turn_id == turn_id:
            return {
                "current": {
                    "answer": turn.answer,
                    "confidence": turn.confidence,
                    "timestamp": turn.timestamp
                },
                "history": turn.regenerated_answers
            }
    
    raise HTTPException(status_code=404, detail="对话轮次不存在")


# ============================================================
# 反思API
# ============================================================

@app.get("/api/conversations/{conversation_id}/reflection")
async def get_reflection(conversation_id: str, user=Depends(get_current_user)):
    """获取对话反思"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 如果已有反思，直接返回
    if conv.reflection:
        return conv.reflection.to_dict()
    
    # 生成新的反思
    reflection = reflector.reflect_on_conversation(conv)
    memory_store.add_reflection(conversation_id, reflection)
    
    return reflection.to_dict()


@app.post("/api/conversations/{conversation_id}/reflection/refresh")
async def refresh_reflection(conversation_id: str, user=Depends(get_current_user)):
    """刷新对话反思"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 生成新的反思
    reflection = reflector.reflect_on_conversation(conv)
    memory_store.add_reflection(conversation_id, reflection)
    
    return reflection.to_dict()


# ============================================================
# 记忆库API
# ============================================================

@app.get("/api/memory/similar")
async def find_similar(
    question: str = Query(..., min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
    user=Depends(get_current_user)
):
    """查找相似问题"""
    similar = memory_store.find_similar_questions(question, user.user_id, top_k)
    
    return [
        {
            "question": turn.question,
            "answer": turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer,
            "confidence": turn.confidence,
            "similarity": score,
            "timestamp": turn.timestamp
        }
        for turn, score in similar
        if score > 0.3  # 只返回相似度较高的
    ]


# ============================================================
# 文档管理API
# ============================================================

@app.post("/api/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    conversation_id: Optional[str] = Form(None),
    user=Depends(get_current_user)
):
    """上传文档（支持最多50个文件）- 后台异步处理"""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="最多支持50个文件")
    
    # 获取或创建对话
    if conversation_id:
        conv = memory_store.get_conversation(conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conv = memory_store.create_conversation(user.user_id)
        conversation_id = conv.conversation_id
    
    # 获取或创建文档会话
    doc_session = document_manager.get_session_by_conversation(conversation_id)
    if not doc_session:
        doc_session = document_manager.create_session(conversation_id, user.user_id)
    
    uploaded = []
    errors = []
    
    for file in files:
        content = await file.read()
        doc = document_manager.add_document_from_file(
            doc_session.session_id, 
            file.filename, 
            content
        )
        if doc:
            uploaded.append(doc.to_dict())
        else:
            errors.append(f"上传失败: {file.filename}")
    
    # 后台异步处理文档（不阻塞响应）
    if uploaded:
        processing_tasks[doc_session.session_id] = {
            "status": "processing",
            "total": len(uploaded),
            "completed": 0,
            "started_at": datetime.now().isoformat()
        }
        background_tasks.add_task(process_documents_background, doc_session.session_id)
    
    return {
        "conversation_id": conversation_id,
        "session_id": doc_session.session_id,
        "uploaded": uploaded,
        "errors": errors,
        "total": len(uploaded),
        "processing": True  # 标记正在后台处理
    }


@app.post("/api/documents/upload-urls")
async def upload_urls(
    data: UrlUploadRequest, 
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """从URL上传文档 - 后台异步处理"""
    if len(data.urls) > 50:
        raise HTTPException(status_code=400, detail="最多支持50个URL")
    
    # 获取或创建对话
    if data.conversation_id:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
        conversation_id = data.conversation_id
    else:
        conv = memory_store.create_conversation(user.user_id)
        conversation_id = conv.conversation_id
    
    # 获取或创建文档会话
    doc_session = document_manager.get_session_by_conversation(conversation_id)
    if not doc_session:
        doc_session = document_manager.create_session(conversation_id, user.user_id)
    
    uploaded = []
    errors = []
    
    for url in data.urls:
        doc = document_manager.add_document_from_url(doc_session.session_id, url)
        if doc:
            uploaded.append(doc.to_dict())
        else:
            errors.append(f"无效URL: {url}")
    
    # 后台异步下载和处理文档
    if uploaded:
        processing_tasks[doc_session.session_id] = {
            "status": "processing",
            "total": len(uploaded),
            "completed": 0,
            "started_at": datetime.now().isoformat()
        }
        background_tasks.add_task(process_url_documents_background, doc_session.session_id)
    
    return {
        "conversation_id": conversation_id,
        "session_id": doc_session.session_id,
        "uploaded": uploaded,
        "errors": errors,
        "total": len(uploaded),
        "processing": True
    }


def parse_single_document(doc, session_id: str):
    """解析单个文档（在线程池中执行）"""
    try:
        document_manager.update_document_status(
            session_id, doc.doc_id, "processing"
        )
        
        parsed = pdf_parser.parse(doc.file_path)
        chunks = pdf_parser.to_chunks(parsed)
        
        # 添加会话信息到chunks，并使用DocumentManager的doc_id替换parser生成的doc_id
        # 这样引用链接才能正确定位到用户上传的文档
        for chunk in chunks:
            chunk["session_id"] = session_id
            chunk["file_path"] = doc.file_path
            chunk["doc_id"] = doc.doc_id  # 使用DocumentManager的doc_id
        
        # 同时更新解析缓存文件名，以便后续能通过doc_id找到缓存
        cache_path = pdf_parser.output_dir / f"{parsed.doc_id}.json"
        new_cache_path = pdf_parser.output_dir / f"{doc.doc_id}.json"
        if cache_path.exists() and not new_cache_path.exists():
            import shutil
            shutil.copy(str(cache_path), str(new_cache_path))
        
        document_manager.update_document_status(
            session_id, doc.doc_id, "ready",
            page_count=len(parsed.pages),
            chunk_count=len(chunks)
        )
        
        return chunks
    except Exception as e:
        document_manager.update_document_status(
            session_id, doc.doc_id, "error",
            error_message=str(e)
        )
        return []


def process_documents_background(session_id: str):
    """后台处理文档（并行解析 + 增量索引）"""
    session = document_manager.get_session(session_id)
    if not session:
        return
    
    pending_docs = [doc for doc in session.documents if doc.status == "pending"]
    if not pending_docs:
        return
    
    all_chunks = []
    completed = 0
    
    # 并行解析文档
    futures = []
    for doc in pending_docs:
        future = executor.submit(parse_single_document, doc, session_id)
        futures.append(future)
    
    # 收集结果
    for future in futures:
        try:
            chunks = future.result(timeout=300)  # 5分钟超时
            all_chunks.extend(chunks)
            completed += 1
            
            # 更新进度
            if session_id in processing_tasks:
                processing_tasks[session_id]["completed"] = completed
        except Exception as e:
            print(f"[Error] 文档解析失败: {e}")
    
    # 增量索引文档
    if all_chunks:
        pipeline.vector_db.index_documents(all_chunks, incremental=True)
    
    # 生成推荐问题
    if all_chunks:
        try:
            questions = report_generator.generate_questions(all_chunks[:20])
            document_manager.set_suggested_questions(session_id, questions)
        except Exception as e:
            print(f"[Warning] 生成推荐问题失败: {e}")
    
    # 标记完成
    if session_id in processing_tasks:
        processing_tasks[session_id]["status"] = "completed"
        processing_tasks[session_id]["completed_at"] = datetime.now().isoformat()


def process_url_documents_background(session_id: str):
    """后台处理URL文档（下载 + 解析 + 索引）"""
    session = document_manager.get_session(session_id)
    if not session:
        return
    
    pending_docs = [doc for doc in session.documents if doc.status == "pending"]
    
    # 先并行下载
    def download_doc(doc):
        return document_manager.download_url_document(doc)
    
    download_futures = [executor.submit(download_doc, doc) for doc in pending_docs]
    
    # 等待下载完成
    for future in download_futures:
        try:
            future.result(timeout=120)  # 2分钟下载超时
        except Exception as e:
            print(f"[Error] 文档下载失败: {e}")
    
    # 然后解析和索引
    process_documents_background(session_id)


async def process_documents(session_id: str):
    """处理文档（兼容旧接口，改为后台处理）"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, process_documents_background, session_id)


@app.get("/api/documents/session/{conversation_id}")
async def get_document_session(conversation_id: str, user=Depends(get_current_user)):
    """获取对话的文档会话"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    session = document_manager.get_session_by_conversation(conversation_id)
    if not session:
        return {"documents": [], "suggested_questions": []}
    
    # 获取处理状态
    processing_status = processing_tasks.get(session.session_id)
    
    return {
        "session_id": session.session_id,
        "documents": [d.to_dict() for d in session.documents],
        "suggested_questions": session.suggested_questions,
        "created_at": session.created_at,
        "processing_status": processing_status
    }


@app.get("/api/documents/status/{session_id}")
async def get_processing_status(session_id: str, user=Depends(get_current_user)):
    """获取文档处理状态"""
    session = document_manager.get_session(session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取处理任务状态
    task_status = processing_tasks.get(session_id, {"status": "unknown"})
    
    # 获取各文档状态
    doc_statuses = [
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "status": doc.status,
            "error_message": doc.error_message
        }
        for doc in session.documents
    ]
    
    return {
        "session_id": session_id,
        "task_status": task_status,
        "documents": doc_statuses,
        "ready_count": sum(1 for d in session.documents if d.status == "ready"),
        "total_count": len(session.documents)
    }


@app.delete("/api/documents/{session_id}/{doc_id}")
async def remove_document(session_id: str, doc_id: str, user=Depends(get_current_user)):
    """移除文档"""
    session = document_manager.get_session(session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    success = document_manager.remove_document(session_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    return {"message": "删除成功"}


@app.post("/api/documents/index")
async def build_index(user=Depends(get_current_user)):
    """构建文档索引"""
    count = pipeline.build_index()
    return {"message": f"索引构建完成，共 {count} 个文档块"}


# ============================================================
# 综述报告API
# ============================================================

@app.post("/api/report/generate")
async def generate_report(data: ReportRequest, user=Depends(get_current_user)):
    """生成综述报告"""
    conv = memory_store.get_conversation(data.conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 获取文档会话
    doc_session = document_manager.get_session_by_conversation(data.conversation_id)
    if not doc_session or not doc_session.documents:
        raise HTTPException(status_code=400, detail="没有可用的文档")
    
    # 检索所有文档的证据
    context = AgentContext(question=data.requirement)
    
    # 从所有文档中检索
    retriever = pipeline._get_agent("retriever")
    retriever_result = retriever.run(context)
    
    evidences = []
    if retriever_result.success:
        evidences = retriever_result.data.get("evidences", [])
    
    if not evidences:
        raise HTTPException(status_code=400, detail="未能从文档中提取有效内容")
    
    # 生成报告
    context.evidences = evidences
    result = report_generator.run(context)
    
    if not result.success:
        raise HTTPException(status_code=500, detail="报告生成失败")
    
    # 保存报告到历史记录
    report_id = secrets.token_hex(16)
    generated_report = GeneratedReport(
        report_id=report_id,
        conversation_id=data.conversation_id,
        user_id=user.user_id,
        requirement=data.requirement,
        report=result.data.get("report", {}),
        citations=result.data.get("citations", []),
        confidence=result.data.get("confidence", 0.0)
    )
    memory_store.save_report(generated_report)
    
    # 返回结果，包含report_id
    response_data = result.data
    response_data["report_id"] = report_id
    return response_data


@app.get("/api/report/history/{conversation_id}")
async def get_report_history(conversation_id: str, user=Depends(get_current_user)):
    """获取对话的历史报告列表"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    reports = memory_store.get_conversation_reports(conversation_id)
    return {
        "reports": [
            {
                "report_id": r.report_id,
                "requirement": r.requirement,
                "title": r.report.get("title", "未命名报告"),
                "confidence": r.confidence,
                "created_at": r.created_at
            }
            for r in sorted(reports, key=lambda x: x.created_at, reverse=True)
        ]
    }


@app.get("/api/report/{report_id}")
async def get_report_detail(report_id: str, user=Depends(get_current_user)):
    """获取报告详情"""
    report = memory_store.get_report(report_id)
    if not report or report.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="报告不存在")
    
    return report.to_dict()


@app.delete("/api/report/{report_id}")
async def delete_report(report_id: str, user=Depends(get_current_user)):
    """删除报告"""
    report = memory_store.get_report(report_id)
    if not report or report.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="报告不存在")
    
    memory_store.delete_report(report_id)
    return {"message": "删除成功"}


@app.get("/api/report/user/all")
async def get_all_user_reports(user=Depends(get_current_user)):
    """获取用户所有历史报告"""
    reports = memory_store.get_user_reports(user.user_id)
    return {
        "reports": [
            {
                "report_id": r.report_id,
                "conversation_id": r.conversation_id,
                "requirement": r.requirement,
                "title": r.report.get("title", "未命名报告"),
                "confidence": r.confidence,
                "created_at": r.created_at
            }
            for r in sorted(reports, key=lambda x: x.created_at, reverse=True)
        ]
    }


@app.get("/api/report/suggested-questions/{conversation_id}")
async def get_suggested_questions(conversation_id: str, user=Depends(get_current_user)):
    """获取推荐问题"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    session = document_manager.get_session_by_conversation(conversation_id)
    if not session:
        return {"questions": []}
    
    return {"questions": session.suggested_questions}


# ============================================================
# 思维导图API
# ============================================================

@app.post("/api/mindmap/generate")
async def generate_mindmap(data: MindmapRequest, user=Depends(get_current_user)):
    """根据文档内容生成思维导图数据"""
    conv = memory_store.get_conversation(data.conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    doc_session = document_manager.get_session_by_conversation(data.conversation_id)
    if not doc_session or not doc_session.documents:
        raise HTTPException(status_code=400, detail="没有可用的文档")
    
    # 筛选文档
    target_docs = doc_session.documents
    if data.doc_ids:
        target_docs = [d for d in doc_session.documents if d.doc_id in data.doc_ids]
    
    if not target_docs:
        raise HTTPException(status_code=400, detail="未找到指定的文档")
    
    # 收集文档内容
    doc_contents = []
    for doc in target_docs:
        if doc.status != "ready":
            continue
        cache_path = pdf_parser.output_dir / f"{doc.doc_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                    text_content = ""
                    for page in parsed.get("pages", [])[:10]:  # 取前10页
                        text_content += page.get("text", "") + "\n"
                    doc_contents.append({
                        "filename": doc.filename,
                        "content": text_content[:5000]  # 限制长度
                    })
            except:
                pass
    
    if not doc_contents:
        raise HTTPException(status_code=400, detail="无法读取文档内容")
    
    # 使用LLM生成思维导图结构
    from ..tools.llm_client import Message
    
    mindmap_prompt = """请根据以下文档内容，生成一个思维导图的JSON结构。
思维导图应该帮助读者快速了解文章的重要内容。

输出格式（JSON）：
{
    "title": "文档主题",
    "children": [
        {
            "name": "主要概念1",
            "children": [
                {"name": "子概念1.1"},
                {"name": "子概念1.2"}
            ]
        },
        {
            "name": "主要概念2",
            "children": [
                {"name": "子概念2.1"},
                {"name": "子概念2.2"}
            ]
        }
    ]
}

请确保：
1. 提取文档的核心主题和关键概念
2. 层次结构清晰，不超过3层
3. 每个节点名称简洁明了
4. 涵盖文档的主要内容"""

    content_text = "\n\n".join([f"文档: {d['filename']}\n{d['content']}" for d in doc_contents])
    
    messages = [
        Message(role="system", content=mindmap_prompt),
        Message(role="user", content=f"请为以下文档生成思维导图：\n\n{content_text[:8000]}")
    ]
    
    try:
        response = pipeline.llm_client.chat(messages, temperature=0.3)
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            mindmap_data = json.loads(json_match.group())
            return {"mindmap": mindmap_data, "documents": [d.filename for d in target_docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成思维导图失败: {str(e)}")
    
    raise HTTPException(status_code=500, detail="无法解析思维导图数据")


# ============================================================
# 翻译API
# ============================================================

@app.post("/api/translate/text")
async def translate_text(data: TranslateRequest, user=Depends(get_current_user)):
    """翻译文本"""
    from ..tools.llm_client import Message
    
    lang_map = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韩文",
        "fr": "法文",
        "de": "德文",
        "es": "西班牙文"
    }
    
    target_lang = lang_map.get(data.target_language, data.target_language)
    
    translate_prompt = f"""请将以下文本翻译成{target_lang}。
要求：
1. 保持原文的专业术语准确性
2. 保持原文的格式和段落结构
3. 翻译要流畅自然
4. 只输出翻译结果，不要添加任何解释

原文：
{data.text}"""
    
    messages = [
        Message(role="user", content=translate_prompt)
    ]
    
    try:
        print(f"[Info] 开始翻译文本，目标语言: {target_lang}，文本长度: {len(data.text)}")
        response = pipeline.llm_client.chat(messages, temperature=0.1)
        
        # 检查响应是否有效
        if not response.content or response.content.startswith("[API调用失败"):
            print(f"[Error] 翻译API返回无效响应: {response.content}")
            raise HTTPException(status_code=500, detail=f"翻译失败: LLM返回无效响应 - {response.content}")
        
        print(f"[Info] 翻译成功，结果长度: {len(response.content)}")
        return {
            "original": data.text,
            "translated": response.content,
            "target_language": data.target_language
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Error] 翻译失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")


@app.post("/api/translate/document")
async def translate_document(data: DocumentTranslateRequest, user=Depends(get_current_user)):
    """翻译整个文档"""
    conv = memory_store.get_conversation(data.conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    doc_session = document_manager.get_session_by_conversation(data.conversation_id)
    if not doc_session:
        raise HTTPException(status_code=404, detail="文档会话不存在")
    
    # 找到目标文档
    target_doc = None
    for doc in doc_session.documents:
        if doc.doc_id == data.doc_id:
            target_doc = doc
            break
    
    if not target_doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    if target_doc.status != "ready":
        raise HTTPException(status_code=400, detail=f"文档尚未处理完成，当前状态: {target_doc.status}")
    
    # 读取文档内容
    cache_path = pdf_parser.output_dir / f"{data.doc_id}.json"
    if not cache_path.exists():
        # 尝试使用原始doc_id查找
        print(f"[Warning] 缓存文件不存在: {cache_path}")
        raise HTTPException(status_code=404, detail="文档内容缓存不存在，请重新上传文档")
    
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    except Exception as e:
        print(f"[Error] 读取文档缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取文档失败: {str(e)}")
    
    pages = parsed.get("pages", [])
    if not pages:
        raise HTTPException(status_code=400, detail="文档内容为空")
    
    from ..tools.llm_client import Message
    
    lang_map = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韩文",
        "fr": "法文",
        "de": "德文",
        "es": "西班牙文"
    }
    target_lang = lang_map.get(data.target_language, data.target_language)
    
    # 处理页数范围
    total_pages = len(pages)
    start_page = data.start_page if data.start_page and data.start_page >= 1 else 1
    end_page = data.end_page if data.end_page and data.end_page <= total_pages else total_pages
    
    # 确保范围有效
    if start_page > end_page:
        start_page, end_page = end_page, start_page
    
    print(f"[Info] 翻译页数范围: {start_page} - {end_page} (共 {total_pages} 页)")
    
    # 筛选要翻译的页面
    pages_to_translate = [p for p in pages if start_page <= p.get("page_num", 0) <= end_page]
    
    # 逐页翻译
    translated_pages = []
    success_count = 0
    
    # LaTeX 格式保留的翻译提示
    latex_instruction = """
5. **重要**：保留所有数学公式的LaTeX格式
   - 行内公式保持 $...$ 格式，如 $E=mc^2$
   - 块级公式保持 $$...$$ 格式
   - 不要翻译公式内容，只翻译公式周围的文字
   - 保留所有LaTeX命令如 \\frac{}{}, \\sum, \\int 等
6. 对于专业术语，可以在翻译后用括号标注英文原文，如：注意力机制(Attention Mechanism)
7. 保持标题、列表等格式结构""" if data.preserve_latex else ""
    
    for page in pages_to_translate:
        page_num = page.get("page_num", 0)
        page_text = page.get("text", "")
        
        if not page_text.strip():
            translated_pages.append({
                "page_num": page_num,
                "original": "",
                "translated": "",
                "has_latex": False
            })
            continue
        
        # 限制单页文本长度，避免超出token限制
        text_to_translate = page_text[:4000]
        
        # 检测是否包含LaTeX公式
        has_latex = bool(re.search(r'\$[^$]+\$|\\\[|\\\(|\\frac|\\sum|\\int', text_to_translate))
        
        translate_prompt = f"""请将以下学术文档内容翻译成{target_lang}。

要求：
1. 保持专业术语的准确性
2. 保持原文的段落结构
3. 翻译要流畅自然
4. 只输出翻译结果，不要添加任何解释或注释{latex_instruction}

原文：
{text_to_translate}"""
        
        messages = [Message(role="user", content=translate_prompt)]
        
        try:
            print(f"[Info] 正在翻译第 {page_num}/{total_pages} 页 (LaTeX: {'是' if has_latex else '否'})...")
            response = pipeline.llm_client.chat(messages, temperature=0.1)
            
            # 检查响应是否有效
            if not response.content or response.content.startswith("[API调用失败"):
                print(f"[Warning] 第{page_num}页翻译API返回无效响应: {response.content}")
                translated_pages.append({
                    "page_num": page_num,
                    "original": page_text,
                    "translated": f"[翻译失败: {response.content}]",
                    "has_latex": has_latex
                })
            else:
                translated_pages.append({
                    "page_num": page_num,
                    "original": page_text,
                    "translated": response.content,
                    "has_latex": has_latex
                })
                success_count += 1
                print(f"[Info] 第{page_num}页翻译成功")
        except Exception as e:
            import traceback
            print(f"[Warning] 第{page_num}页翻译失败: {e}")
            traceback.print_exc()
            translated_pages.append({
                "page_num": page_num,
                "original": page_text,
                "translated": f"[翻译失败: {str(e)}]",
                "has_latex": has_latex
            })
    
    if success_count == 0 and len(pages_to_translate) > 0:
        raise HTTPException(status_code=500, detail="所有页面翻译均失败，请检查LLM服务是否正常")
    
    return {
        "doc_id": data.doc_id,
        "filename": target_doc.filename,
        "target_language": data.target_language,
        "pages": translated_pages,
        "total_pages": total_pages,
        "translated_range": {"start": start_page, "end": end_page},
        "success_count": success_count,
        "preserve_latex": data.preserve_latex
    }


# ============================================================
# 感兴趣内容推荐API
# ============================================================

@app.post("/api/interesting/generate")
async def generate_interesting_content(data: InterestingContentRequest, user=Depends(get_current_user)):
    """根据上传文档生成用户可能感兴趣的内容"""
    conv = memory_store.get_conversation(data.conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    doc_session = document_manager.get_session_by_conversation(data.conversation_id)
    if not doc_session or not doc_session.documents:
        raise HTTPException(status_code=400, detail="没有可用的文档")
    
    # 收集文档摘要
    doc_summaries = []
    for doc in doc_session.documents:
        if doc.status != "ready":
            continue
        cache_path = pdf_parser.output_dir / f"{doc.doc_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                    # 取前两页作为摘要
                    summary = ""
                    for page in parsed.get("pages", [])[:2]:
                        summary += page.get("text", "")[:1000]
                    doc_summaries.append({
                        "filename": doc.filename,
                        "summary": summary
                    })
            except:
                pass
    
    if not doc_summaries:
        raise HTTPException(status_code=400, detail="无法读取文档内容")
    
    # 获取用户历史反思上下文
    reflection_context = memory_store.get_reflection_context_for_conversation(
        user.user_id, ""
    )
    
    from ..tools.llm_client import Message
    
    interest_prompt = """基于用户上传的文档内容，分析并推荐用户可能感兴趣的内容。

请输出JSON格式：
{
    "main_topics": ["主题1", "主题2", "主题3"],
    "key_findings": [
        {"title": "发现1标题", "description": "简要描述"},
        {"title": "发现2标题", "description": "简要描述"}
    ],
    "related_questions": [
        "用户可能想问的问题1",
        "用户可能想问的问题2",
        "用户可能想问的问题3"
    ],
    "recommended_explorations": [
        {"topic": "探索方向1", "reason": "推荐理由"},
        {"topic": "探索方向2", "reason": "推荐理由"}
    ],
    "summary": "文档内容的整体概述"
}"""

    content_text = "\n\n".join([f"文档: {d['filename']}\n{d['summary']}" for d in doc_summaries])
    
    # 加入历史偏好
    preference_text = ""
    if reflection_context.get("user_preferences"):
        preference_text = f"\n\n用户历史偏好: {json.dumps(reflection_context['user_preferences'], ensure_ascii=False)}"
    
    messages = [
        Message(role="system", content=interest_prompt),
        Message(role="user", content=f"请分析以下文档并推荐感兴趣的内容：\n\n{content_text[:6000]}{preference_text}")
    ]
    
    try:
        response = pipeline.llm_client.chat(messages, temperature=0.5)
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            interest_data = json.loads(json_match.group())
            return {"interesting_content": interest_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成推荐内容失败: {str(e)}")
    
    raise HTTPException(status_code=500, detail="无法解析推荐内容")


# ============================================================
# 图片可视化生成API
# ============================================================

@app.post("/api/visualization/generate")
async def generate_visualization(data: ImageVisualizationRequest, user=Depends(get_current_user)):
    """根据论文内容和用户提示生成可视化图片"""
    try:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        doc_session = document_manager.get_session_by_conversation(data.conversation_id)
        if not doc_session or not doc_session.documents:
            raise HTTPException(status_code=400, detail="没有可用的文档")
        
        # 找到目标文档
        target_doc = None
        for doc in doc_session.documents:
            if doc.doc_id == data.doc_id:
                target_doc = doc
                break
        
        if not target_doc or target_doc.status != "ready":
            raise HTTPException(status_code=404, detail="文档不存在或未处理完成")
        
        # 检索文档相关证据
        context = AgentContext(question=data.prompt)
        retriever = pipeline._get_agent("retriever")
        retriever_result = retriever.run(context)
        
        evidences = []
        if retriever_result.success:
            # 过滤只保留目标文档的证据
            all_evidences = retriever_result.data.get("evidences", [])
            evidences = [ev for ev in all_evidences if ev.get("doc_id") == data.doc_id]
            # 如果过滤后太少，也包含一些其他相关证据
            if len(evidences) < 5:
                evidences = all_evidences[:20]
        
        if not evidences:
            raise HTTPException(status_code=400, detail="未能从文档中提取有效内容")
        
        # 生成可视化图片
        print(f"[Info] 开始生成图片，提示词: {data.prompt[:50]}...")
        context.evidences = evidences
        result = image_generator.run(context)
        
        if not result.success:
            print(f"[Error] 图片生成失败: {result.error}")
            raise HTTPException(status_code=500, detail=f"图片生成失败: {result.error}")
        
        # 获取生成结果
        viz_data = result.data.get("visualization", {})
        image_result = result.data.get("image_result", {})
        prompt_info = result.data.get("prompt_info", {})
        
        # 检查图片是否生成成功
        if not image_result.get("success"):
            error_msg = image_result.get("error", "图片生成失败")
            print(f"[Error] 图片API返回错误: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"[Info] 图片生成成功")
        return {
            "doc_id": data.doc_id,
            "filename": target_doc.filename,
            "prompt": data.prompt,
            "visualization": {
                "title": viz_data.get("title", prompt_info.get("title_zh", "生成的图片")),
                "description": viz_data.get("description", prompt_info.get("description_zh", "")),
                "style": viz_data.get("style", prompt_info.get("style", "diagram")),
                "image_url": image_result.get("image_url"),
                "image_base64": image_result.get("image_base64"),
                "image_path": image_result.get("image_path"),
                "prompt_used": image_result.get("prompt_used", prompt_info.get("image_prompt", "")),
                "generated_at": viz_data.get("generated_at")
            },
            "note": "图片已生成，请确保内容与论文相关"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Error] 图片生成异常: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/visualization/upload-image")
async def upload_image_for_visualization(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    doc_id: str = Form(...),
    prompt: str = Form(...),
    user=Depends(get_current_user)
):
    """上传图片进行可视化分析（最多1个图片）"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 验证文件类型
    allowed_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式，支持: {allowed_extensions}")
    
    # 验证文件大小（10MB）
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过10MB")
    
    # 保存临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 检索文档相关证据
        context = AgentContext(question=prompt)
        retriever = pipeline._get_agent("retriever")
        retriever_result = retriever.run(context)
        
        evidences = []
        if retriever_result.success:
            all_evidences = retriever_result.data.get("evidences", [])
            evidences = [ev for ev in all_evidences if ev.get("doc_id") == doc_id]
            if len(evidences) < 5:
                evidences = all_evidences[:20]
        
        # 基于图片和文档生成可视化
        result = image_generator.generate_visualization_from_image(
            tmp_path, prompt, evidences
        )
        
        return {
            "doc_id": doc_id,
            "prompt": prompt,
            "visualization": result,
            "note": "请确保上传的图片和提示词与选中的论文内容相关"
        }
    finally:
        # 清理临时文件
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/visualization/document-list/{conversation_id}")
async def get_documents_for_visualization(conversation_id: str, user=Depends(get_current_user)):
    """获取可用于可视化的文档列表"""
    conv = memory_store.get_conversation(conversation_id)
    if not conv or conv.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    doc_session = document_manager.get_session_by_conversation(conversation_id)
    if not doc_session:
        return {"documents": []}
    
    ready_docs = [
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "page_count": doc.page_count
        }
        for doc in doc_session.documents
        if doc.status == "ready"
    ]
    
    return {
        "documents": ready_docs,
        "note": "请选择一个文档，并输入与该文档内容相关的提示词"
    }


# ============================================================
# 多LLM推理API
# ============================================================

@app.post("/api/reasoning/multi-llm")
async def multi_llm_reasoning(data: QuestionRequest, user=Depends(get_current_user)):
    """使用多LLM协同推理（支持文本推理、公式推理等）"""
    # 获取或创建对话
    if data.conversation_id:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conv = memory_store.create_conversation(user.user_id)
    
    # 检索证据
    context = AgentContext(question=data.question)
    retriever = pipeline._get_agent("retriever")
    retriever_result = retriever.run(context)
    
    evidences = []
    if retriever_result.success:
        evidences = retriever_result.data.get("evidences", [])
    
    # 执行多LLM推理
    context.evidences = evidences
    result = multi_llm_reasoner.run(context)
    
    if not result.success:
        raise HTTPException(status_code=500, detail="推理失败")
    
    data_result = result.data
    
    # 创建对话轮次
    turn_id = secrets.token_hex(16)
    turn = ConversationTurn(
        turn_id=turn_id,
        question=data.question,
        answer=data_result.get("answer", ""),
        reasoning_trace=data_result.get("reasoning_trace"),
        confidence=data_result.get("confidence", 0.5),
        citations=[]
    )
    memory_store.add_turn(conv.conversation_id, turn)
    
    return {
        "turn_id": turn_id,
        "conversation_id": conv.conversation_id,
        "answer": data_result.get("answer", ""),
        "confidence": data_result.get("confidence", 0.5),
        "analysis_type": data_result.get("analysis_type", "text"),
        "formulas": data_result.get("formulas", []),
        "reasoning_trace": data_result.get("reasoning_trace").to_dict() if hasattr(data_result.get("reasoning_trace"), "to_dict") else data_result.get("reasoning_trace"),
        "math_analysis": data_result.get("math_analysis")
    }


@app.post("/api/reasoning/multi-llm/stream")
async def multi_llm_reasoning_stream(data: QuestionRequest, user=Depends(get_current_user)):
    """流式多LLM推理"""
    # 获取或创建对话
    if data.conversation_id:
        conv = memory_store.get_conversation(data.conversation_id)
        if not conv or conv.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conv = memory_store.create_conversation(user.user_id)
    
    async def generate():
        yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关文档...'})}\n\n"
        
        # 检索证据
        context = AgentContext(question=data.question)
        retriever = pipeline._get_agent("retriever")
        retriever_result = retriever.run(context)
        
        evidences = []
        if retriever_result.success:
            evidences = retriever_result.data.get("evidences", [])
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'检索到 {len(evidences)} 条相关证据'})}\n\n"
        
        # 流式输出推理过程
        for step in multi_llm_reasoner.reason_stream(data.question, evidences):
            yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
        
        # 保存对话轮次
        turn_id = secrets.token_hex(16)
        # 注意：流式输出时，最终结果在最后一个step中
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# 反思上下文API（用于下次对话）
# ============================================================

@app.get("/api/reflection/context")
async def get_reflection_context(user=Depends(get_current_user)):
    """获取用户的反思上下文，用于增强下次对话"""
    context = memory_store.get_reflection_context_for_conversation(user.user_id, "")
    return context


# ============================================================
# 历史记忆API
# ============================================================

@app.get("/api/history/context")
async def get_history_context(
    question: str = Query(..., min_length=1),
    user=Depends(get_current_user)
):
    """获取用户历史上下文（用于跨对话记忆）"""
    context = memory_store.get_user_history_context(user.user_id, question)
    return context


@app.get("/api/history/documents")
async def get_user_documents_cache(user=Depends(get_current_user)):
    """获取用户所有引用过的文档缓存"""
    documents = memory_store.get_all_user_documents_cache(user.user_id)
    return {"documents": documents}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ============================================================
# 生成图片API
# ============================================================

@app.get("/api/generated-image/{image_id}")
async def get_generated_image(image_id: str, user=Depends(get_current_user)):
    """获取生成的图片文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = Path(base_dir) / "data" / "generated_images"
    
    # 查找图片文件
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        image_path = image_dir / f"generated_{image_id}{ext}"
        if image_path.exists():
            return FileResponse(
                path=str(image_path),
                media_type=f"image/{ext[1:]}",
                filename=f"generated_{image_id}{ext}"
            )
    
    raise HTTPException(status_code=404, detail="图片不存在")


# ============================================================
# PDF预览API
# ============================================================

@app.get("/api/pdf/{doc_id}")
async def get_pdf_file(doc_id: str, user=Depends(get_current_user)):
    """获取PDF文件"""
    # 检查 doc_id 是否为空
    if not doc_id or doc_id.strip() == "":
        raise HTTPException(status_code=400, detail="无效的文档ID")
    
    # 在所有用户的文档会话中查找该文档
    sessions = document_manager.get_user_sessions(user.user_id)
    
    for session in sessions:
        for doc in session.documents:
            if doc.doc_id == doc_id:
                file_path = Path(doc.file_path)
                if file_path.exists() and file_path.suffix.lower() == '.pdf':
                    return FileResponse(
                        path=str(file_path),
                        media_type="application/pdf",
                        filename=doc.filename
                    )
                raise HTTPException(status_code=404, detail="PDF文件不存在")
    
    # 回退：通过向量数据库中的doc_id查找文件路径
    # 这是为了兼容旧的索引数据（使用pdf_parser生成的doc_id）
    for doc_meta in pipeline.vector_db.docs:
        if doc_meta.get("doc_id") == doc_id:
            source_path = doc_meta.get("source", "")
            file_path = doc_meta.get("file_path", source_path)
            if file_path:
                p = Path(file_path)
                if p.exists() and p.suffix.lower() == '.pdf':
                    return FileResponse(
                        path=str(p),
                        media_type="application/pdf",
                        filename=p.name
                    )
            break
    
    raise HTTPException(status_code=404, detail="文档不存在")


@app.get("/api/pdf/{doc_id}/info")
async def get_pdf_info(doc_id: str, user=Depends(get_current_user)):
    """获取PDF文档信息"""
    # 检查 doc_id 是否为空
    if not doc_id or doc_id.strip() == "":
        raise HTTPException(status_code=400, detail="无效的文档ID")
    
    print(f"[Debug] 查找文档 doc_id={doc_id}, user_id={user.user_id}")
    
    sessions = document_manager.get_user_sessions(user.user_id)
    print(f"[Debug] 用户会话数: {len(sessions)}")
    
    for session in sessions:
        for doc in session.documents:
            if doc.doc_id == doc_id:
                # 尝试从缓存获取解析信息
                cache_path = pdf_parser.output_dir / f"{doc_id}.json"
                page_count = doc.page_count or 0
                
                if cache_path.exists():
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            page_count = len(data.get("pages", []))
                    except:
                        pass
                
                return {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "page_count": page_count,
                    "status": doc.status,
                    "file_path": doc.file_path
                }
    
    # 回退：通过向量数据库中的doc_id查找文件信息
    # 这是为了兼容旧的索引数据（使用pdf_parser生成的doc_id）
    print(f"[Debug] 在用户会话中未找到文档，尝试从向量数据库查找，docs数量: {len(pipeline.vector_db.docs)}")
    for doc_meta in pipeline.vector_db.docs:
        if doc_meta.get("doc_id") == doc_id:
            source_path = doc_meta.get("source", "")
            file_path = doc_meta.get("file_path", source_path)
            filename = Path(file_path).name if file_path else "unknown.pdf"
            
            # 尝试从缓存获取页数
            cache_path = pdf_parser.output_dir / f"{doc_id}.json"
            page_count = 0
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        page_count = len(data.get("pages", []))
                except:
                    pass
            
            return {
                "doc_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "status": "ready",
                "file_path": file_path
            }
    
    print(f"[Debug] 文档未找到: doc_id={doc_id}")
    raise HTTPException(status_code=404, detail="文档不存在")


@app.get("/api/pdf/{doc_id}/page/{page_num}")
async def get_pdf_page_content(doc_id: str, page_num: int, user=Depends(get_current_user)):
    """获取PDF指定页面的解析内容"""
    # 检查 doc_id 是否为空
    if not doc_id or doc_id.strip() == "":
        raise HTTPException(status_code=400, detail="无效的文档ID")
    
    # 辅助函数：从缓存读取页面内容
    def read_page_from_cache(cache_path: Path, page_num: int):
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    pages = data.get("pages", [])
                    for page in pages:
                        if page.get("page_num") == page_num:
                            return {
                                "page_num": page_num,
                                "text": page.get("text", ""),
                                "tables": page.get("tables", []),
                                "images": page.get("images", [])
                            }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取页面内容失败: {str(e)}")
        return None
    
    sessions = document_manager.get_user_sessions(user.user_id)
    
    for session in sessions:
        for doc in session.documents:
            if doc.doc_id == doc_id:
                # 从缓存获取页面内容
                cache_path = pdf_parser.output_dir / f"{doc_id}.json"
                result = read_page_from_cache(cache_path, page_num)
                if result:
                    return result
                raise HTTPException(status_code=404, detail="页面内容不存在")
    
    # 回退：通过向量数据库中的doc_id查找缓存
    # 这是为了兼容旧的索引数据（使用pdf_parser生成的doc_id）
    for doc_meta in pipeline.vector_db.docs:
        if doc_meta.get("doc_id") == doc_id:
            cache_path = pdf_parser.output_dir / f"{doc_id}.json"
            result = read_page_from_cache(cache_path, page_num)
            if result:
                return result
            raise HTTPException(status_code=404, detail="页面内容不存在")
    
    raise HTTPException(status_code=404, detail="文档不存在")


@app.get("/api/documents/list")
async def list_user_documents(user=Depends(get_current_user)):
    """获取用户所有文档列表"""
    sessions = document_manager.get_user_sessions(user.user_id)
    documents = []
    
    for session in sessions:
        for doc in session.documents:
            documents.append({
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "status": doc.status,
                "page_count": doc.page_count,
                "conversation_id": session.conversation_id,
                "created_at": doc.created_at if hasattr(doc, 'created_at') else None
            })
    
    return {"documents": documents}


# ============================================================
# 静态文件服务
# ============================================================

# 挂载静态文件目录
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
async def serve_index():
    """提供前端页面"""
    index_path = web_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "SciAgent API", "docs": "/docs"}
