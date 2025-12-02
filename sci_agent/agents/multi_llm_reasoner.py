"""
Multi-LLM Reasoner Agent - 多模型协同推理
支持文本推理、公式推理、视觉理解等多种能力的LLM协同工作

整合了原 ReasonerAgent 的功能，提供：
- 基础推理：简单证据合成
- 思维链推理：多步骤推理
- 多LLM协同：文本+数学+视觉
- 流式输出：实时展示推理过程
"""
import re
import json
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator

from .base import BaseAgent, AgentContext, AgentResult
from ..models.memory import ThinkingStep, ReasoningTrace


# ============================================================
# 基础推理 Prompt（兼容原 ReasonerAgent）
# ============================================================

REASONER_SYSTEM_PROMPT = """你是一个专业的科学研究助手。基于提供的证据回答问题。

要求：
1. 答案必须基于提供的证据，不要编造信息
2. 每个关键论点必须标注引用，格式：[来源X]
3. 如果证据不足以回答问题，明确说明
4. 使用专业、准确的语言
5. 数学公式必须使用LaTeX格式：
   - 行内公式：$E=mc^2$
   - 块级公式：$$\\sum_{i=1}^n x_i$$

输出格式（JSON）：
{
    "answer": "完整答案（包含引用标记，数学公式用LaTeX格式如 $x^2$ 或 $$\\\\frac{a}{b}$$）",
    "key_points": ["关键点1", "关键点2"],
    "citations": [
        {"id": 1, "source": "来源", "page": 页码, "quote": "引用原文"},
        {"id": 2, "source": "来源", "page": 页码, "quote": "引用原文"}
    ],
    "confidence_factors": {
        "evidence_quality": 0.8,
        "evidence_coverage": 0.7,
        "reasoning_clarity": 0.9
    },
    "limitations": "答案的局限性说明"
}"""

COT_PROMPT = """你是一个专业的科学研究助手。请使用思维链方法逐步分析问题。

步骤：
1. 理解问题：明确问题的核心要求
2. 分析证据：逐条分析每个证据的相关性
3. 建立联系：找出证据之间的逻辑关系
4. 推理结论：基于证据推导答案
5. 验证答案：检查答案是否有充分支持

**数学公式格式**：使用LaTeX语法
- 行内公式：$公式$（如 $E=mc^2$）
- 块级公式：$$公式$$（如 $$\\sum_{i=1}^n x_i$$）

请按以下格式输出：
{
    "thinking_steps": [
        {"step": 1, "title": "理解问题", "content": "..."},
        {"step": 2, "title": "分析证据", "content": "..."},
        ...
    ],
    "answer": "最终答案（包含引用标记，数学公式用LaTeX）",
    "citations": [...],
    "confidence": 0.85
}"""


MULTI_LLM_SYSTEM_PROMPT = """你是一个专业的科学研究助手，需要展示完整的推理过程。

你的推理过程应该包含：
1. **问题分析**：理解问题的核心需求和类型
2. **证据整理**：分析每条证据的相关性
3. **逻辑推理**：建立从证据到结论的推理链
4. **数学/公式分析**（如适用）：分析涉及的数学公式或定量关系
5. **综合结论**：给出最终答案

**重要：数学公式格式要求**
- 行内公式使用单个美元符号包裹，如：$E=mc^2$、$\\alpha$、$\\frac{a}{b}$
- 块级公式（独立成行）使用双美元符号包裹，如：
  $$\\text{Attn}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
- 所有数学符号、变量、方程都必须使用LaTeX格式
- 希腊字母：$\\alpha$, $\\beta$, $\\gamma$, $\\theta$ 等
- 分数：$\\frac{a}{b}$，根号：$\\sqrt{x}$，求和：$\\sum_{i=1}^n$
- 矩阵、向量使用适当的LaTeX环境

输出格式（JSON）：
{
    "analysis_type": "text|math|mixed",
    "thinking_steps": [
        {"step_id": 1, "title": "问题分析", "content": "...", "reasoning_type": "text"},
        {"step_id": 2, "title": "证据整理", "content": "...", "reasoning_type": "text"},
        {"step_id": 3, "title": "公式推导", "content": "公式使用LaTeX，如 $x^2$ 或 $$\\sum x_i$$", "reasoning_type": "math"},
        {"step_id": 4, "title": "综合结论", "content": "...", "reasoning_type": "text"}
    ],
    "formulas": [
        {"latex": "E=mc^2", "explanation": "质能方程"}
    ],
    "answer": "最终答案（数学公式使用LaTeX格式）",
    "confidence": 0.85
}"""


MATH_REASONING_PROMPT = """你是一个专业的数学和公式推理专家。请分析以下内容中涉及的数学公式、定量关系和推导过程。

要求：
1. 识别文本中的数学公式和定量关系
2. 解释公式的含义和推导过程
3. 如果有计算，展示计算步骤
4. 使用LaTeX格式表示公式

**LaTeX公式格式规范**：
- 行内公式用单美元符号：$E=mc^2$
- 块级公式用双美元符号：$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$
- 常用符号：
  - 分数：$\\frac{a}{b}$
  - 根号：$\\sqrt{x}$, $\\sqrt[n]{x}$
  - 求和：$\\sum_{i=1}^{n} x_i$
  - 积分：$\\int_a^b f(x)dx$
  - 希腊字母：$\\alpha, \\beta, \\gamma, \\theta, \\lambda$
  - 矩阵：使用 \\begin{matrix}...\\end{matrix}
  - 向量：$\\vec{v}$ 或 $\\mathbf{v}$

输出格式（JSON）：
{
    "formulas_found": [
        {"latex": "公式的LaTeX表示（如 \\\\frac{a}{b}）", "explanation": "公式含义", "context": "出现的上下文"}
    ],
    "derivations": [
        {"step": 1, "content": "推导步骤1（公式用LaTeX）"},
        {"step": 2, "content": "推导步骤2（公式用LaTeX）"}
    ],
    "quantitative_analysis": "定量分析结果（公式用LaTeX格式）",
    "summary": "数学分析总结"
}"""


class MultiLLMReasonerAgent(BaseAgent):
    """
    多LLM协同推理Agent（整合了原 ReasonerAgent）
    
    功能：
    - 基础推理：简单证据合成（兼容原 ReasonerAgent）
    - 思维链推理：多步骤推理（兼容原 ChainOfThoughtReasoner）
    - 文本推理：使用通用LLM进行文本理解和推理
    - 数学推理：使用数学专长LLM进行公式分析和推导
    - 视觉理解：使用VL模型进行图像内容理解
    - 协同推理：综合多个模型的输出生成最终结果
    - 流式输出：实时展示推理过程
    """
    
    def __init__(self,
                 config: Dict[str, Any] = None,
                 llm_client = None,  # 兼容原 ReasonerAgent 的参数名
                 text_llm_client = None,
                 math_llm_client = None,
                 vision_llm_client = None):
        super().__init__(name="multi_llm_reasoner", config=config)
        # 兼容两种初始化方式
        self.text_llm = text_llm_client or llm_client
        self.math_llm = math_llm_client or self.text_llm  # 回退到文本LLM
        self.vision_llm = vision_llm_client
        # 兼容原 ReasonerAgent 的属性名
        self.llm_client = self.text_llm
        
        # 读取集成推理配置 - 兼容两种传参方式
        # 方式1: config={"reasoning": {...}}
        # 方式2: config={...reasoning配置直接传入...}
        config = config or {}
        if "reasoning" in config:
            self.reasoning_config = config.get("reasoning", {})
        elif "enable_ensemble" in config or "ensemble_models" in config:
            # 直接传入了 reasoning 配置
            self.reasoning_config = config
        else:
            self.reasoning_config = {}
        
        self.ensemble_models = self.reasoning_config.get("ensemble_models", [])
        self.strategy = self.reasoning_config.get("strategy", {})
        self.enable_ensemble = self.reasoning_config.get("enable_ensemble", False)
        
        # 调试日志
        print(f"[Debug] MultiLLMReasoner 初始化:")
        print(f"  - enable_ensemble: {self.enable_ensemble}")
        print(f"  - ensemble_models: {len(self.ensemble_models)} 个")
        print(f"  - strategy.mode: {self.strategy.get('mode', 'single')}")
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行多LLM协同推理"""
        question = context.question
        evidences = context.evidences
        captions = context.captions
        
        if not evidences:
            print(f"  [MultiLLMReasoner] 未找到证据，返回空结果")
            return AgentResult(
                success=True,
                data={
                    "reasoning_trace": self._create_empty_trace(question),
                    "answer": "未找到相关证据，无法进行分析。",
                    "confidence": 0.0
                }
            )
        
        # 分析问题类型
        analysis_type = self._analyze_question_type(question, evidences)
        print(f"  [MultiLLMReasoner] 问题类型: {analysis_type} ({'数学推理' if analysis_type == 'math' else '混合推理' if analysis_type == 'mixed' else '文本推理'})")
        print(f"  [MultiLLMReasoner] 证据数量: {len(evidences)}")
        
        # 根据类型选择推理策略
        if analysis_type == "math":
            print(f"  [MultiLLMReasoner] 使用数学推理模式...")
            result = self._math_focused_reasoning(question, evidences, captions)
        elif analysis_type == "mixed":
            print(f"  [MultiLLMReasoner] 使用混合推理模式...")
            result = self._mixed_reasoning(question, evidences, captions)
        else:
            print(f"  [MultiLLMReasoner] 使用文本推理模式...")
            result = self._text_reasoning(question, evidences, captions)
        
        print(f"  [MultiLLMReasoner] 推理完成，置信度: {result.get('confidence', 0):.2%}")
        return AgentResult(success=True, data=result)
    
    def _analyze_question_type(self, question: str, evidences: List[Dict]) -> str:
        """分析问题类型"""
        # 检测数学相关关键词
        math_keywords = [
            '公式', '方程', '计算', '推导', '证明', '定理', '算法',
            'formula', 'equation', 'calculate', 'derive', 'theorem',
            '=', '∑', '∫', '∂', 'Σ', '√'
        ]
        
        text = question.lower()
        for ev in evidences[:5]:
            text += " " + ev.get("text", "").lower()
        
        math_score = sum(1 for kw in math_keywords if kw.lower() in text)
        
        if math_score >= 3:
            return "math"
        elif math_score >= 1:
            return "mixed"
        return "text"
    
    def _text_reasoning(self, question: str, evidences: List[Dict], 
                        captions: List[Dict]) -> Dict[str, Any]:
        """纯文本推理 - 支持集成多模型"""
        from ..tools.llm_client import Message, LLMClient
        
        evidence_context = self._build_evidence_context(evidences, captions)
        
        messages = [
            Message(role="system", content=MULTI_LLM_SYSTEM_PROMPT),
            Message(role="user", content=f"问题：{question}\n\n证据：\n{evidence_context}")
        ]
        
        # 检查是否启用集成推理
        strategy_mode = self.strategy.get("mode", "single")
        
        if self.enable_ensemble and self.ensemble_models and strategy_mode == "ensemble":
            print(f"  [MultiLLMReasoner] 启用集成推理模式，调用 {len(self.ensemble_models)} 个模型...")
            return self._ensemble_reasoning(question, evidences, captions, messages)
        
        # 单模型推理
        try:
            print(f"  [MultiLLMReasoner] 调用模型A: {self.text_llm.model} (provider: {self.text_llm.api_provider}, url: {self.text_llm.api_base})")
            response = self.text_llm.chat(messages, temperature=0.3, max_tokens=4096)
            return self._parse_reasoning_response(question, response.content, evidences)
        except Exception as e:
            print(f"[Warning] 文本推理失败: {e}")
            return self._fallback_reasoning(question, evidences, captions)
    
    def _ensemble_reasoning(self, question: str, evidences: List[Dict],
                            captions: List[Dict], messages: List) -> Dict[str, Any]:
        """集成多模型推理"""
        from ..tools.llm_client import LLMClient
        
        results = []
        weights = []
        
        for i, model_cfg in enumerate(self.ensemble_models):
            model_name = model_cfg.get("model", "")
            provider = model_cfg.get("provider", "siliconflow")
            weight = model_cfg.get("weight", 1.0 / len(self.ensemble_models))
            
            print(f"  [MultiLLMReasoner] 调用模型{chr(65+i)}: {model_name} (provider: {provider}, weight: {weight})")
            
            try:
                # 创建临时LLM客户端
                temp_llm = LLMClient(model=model_name, api_provider=provider)
                print(f"    -> API URL: {temp_llm.api_base}")
                
                response = temp_llm.chat(messages, temperature=0.3, max_tokens=4096)
                parsed = self._parse_reasoning_response(question, response.content, evidences)
                
                if parsed.get("answer"):
                    results.append(parsed)
                    weights.append(weight)
                    print(f"    -> 模型{chr(65+i)}回答成功，置信度: {parsed.get('confidence', 0):.2%}")
                else:
                    print(f"    -> 模型{chr(65+i)}返回空答案")
                    
            except Exception as e:
                print(f"    -> 模型{chr(65+i)}调用失败: {e}")
        
        if not results:
            print(f"  [MultiLLMReasoner] 所有集成模型都失败，回退到主模型")
            return self._fallback_reasoning(question, evidences, captions)
        
        # 根据策略整合结果
        ensemble_method = self.strategy.get("ensemble_method", "weighted")
        print(f"  [MultiLLMReasoner] 整合 {len(results)} 个模型结果，方法: {ensemble_method}")
        
        return self._merge_ensemble_results(results, weights, ensemble_method, question, evidences)
    
    def _merge_ensemble_results(self, results: List[Dict], weights: List[float],
                                 method: str, question: str, evidences: List[Dict]) -> Dict[str, Any]:
        """整合多模型结果 - 基于置信度动态加权"""
        if method == "best":
            # 选择置信度最高的
            best_idx = max(range(len(results)), key=lambda i: results[i].get("confidence", 0))
            return results[best_idx]
        
        elif method == "weighted":
            # 动态加权：基于各模型自我校准的置信度计算权重
            # 置信度越高，权重越大
            confidences = [r.get("confidence", 0.5) for r in results]
            
            # 打印各模型置信度
            for i, conf in enumerate(confidences):
                print(f"    -> 模型{chr(65+i)} 置信度: {conf:.2%}")
            
            # 计算动态权重：使用置信度的平方来放大高置信度模型的影响
            # 这样置信度高的模型会获得更大的权重
            confidence_squared = [c ** 2 for c in confidences]
            total_conf_sq = sum(confidence_squared)
            
            if total_conf_sq > 0:
                dynamic_weights = [c_sq / total_conf_sq for c_sq in confidence_squared]
            else:
                # 如果所有置信度都是0，回退到均匀权重
                dynamic_weights = [1.0 / len(results)] * len(results)
            
            # 打印动态权重
            print(f"  [MultiLLMReasoner] 动态权重分配:")
            for i, (orig_w, dyn_w) in enumerate(zip(weights, dynamic_weights)):
                print(f"    -> 模型{chr(65+i)}: 原始权重 {orig_w:.2f} -> 动态权重 {dyn_w:.2%}")
            
            # 计算加权置信度
            weighted_confidence = sum(
                conf * dyn_w for conf, dyn_w in zip(confidences, dynamic_weights)
            )
            
            # 选择动态权重最高的结果作为主答案
            best_idx = max(range(len(results)), key=lambda i: dynamic_weights[i])
            best_result = results[best_idx]
            print(f"  [MultiLLMReasoner] 选择模型{chr(65+best_idx)}作为主答案 (动态权重: {dynamic_weights[best_idx]:.2%})")
            
            # 整合所有模型的推理步骤
            all_steps = []
            for i, r in enumerate(results):
                trace = r.get("reasoning_trace")
                if trace and hasattr(trace, "thinking_steps"):
                    for step in trace.thinking_steps:
                        step.title = f"[模型{chr(65+i)}] {step.title}"
                        all_steps.append(step)
            
            # 更新结果
            best_result["confidence"] = weighted_confidence
            best_result["ensemble_info"] = {
                "model_count": len(results),
                "method": "dynamic_weighted",
                "original_weights": weights,
                "dynamic_weights": dynamic_weights,
                "confidences": confidences
            }
            
            return best_result
        
        else:  # vote
            # 简单投票 - 选择置信度最高的
            best_idx = max(range(len(results)), key=lambda i: results[i].get("confidence", 0))
            return results[best_idx]
    
    def _math_focused_reasoning(self, question: str, evidences: List[Dict],
                                 captions: List[Dict]) -> Dict[str, Any]:
        """数学/公式推理"""
        from ..tools.llm_client import Message
        
        evidence_context = self._build_evidence_context(evidences, captions)
        
        # 先用数学LLM分析公式
        math_messages = [
            Message(role="system", content=MATH_REASONING_PROMPT),
            Message(role="user", content=f"请分析以下内容中的数学公式和定量关系：\n\n{evidence_context}")
        ]
        
        math_analysis = {}
        try:
            print(f"  [MultiLLMReasoner] 调用数学模型: {self.math_llm.model} (provider: {self.math_llm.api_provider}, url: {self.math_llm.api_base})")
            math_response = self.math_llm.chat(math_messages, temperature=0.2, max_tokens=2048)
            json_match = re.search(r'\{[\s\S]*\}', math_response.content)
            if json_match:
                math_analysis = json.loads(json_match.group())
                print(f"  [MultiLLMReasoner] 数学分析完成，发现 {len(math_analysis.get('formulas_found', []))} 个公式")
        except Exception as e:
            print(f"[Warning] 数学分析失败: {e}")
        
        # 再用文本LLM综合推理
        combined_prompt = f"""问题：{question}

证据：
{evidence_context}

数学分析结果：
{json.dumps(math_analysis, ensure_ascii=False, indent=2) if math_analysis else "无"}

请综合以上信息进行推理分析。"""
        
        messages = [
            Message(role="system", content=MULTI_LLM_SYSTEM_PROMPT),
            Message(role="user", content=combined_prompt)
        ]
        
        try:
            print(f"  [MultiLLMReasoner] 调用文本模型整合: {self.text_llm.model} (provider: {self.text_llm.api_provider}, url: {self.text_llm.api_base})")
            response = self.text_llm.chat(messages, temperature=0.3, max_tokens=4096)
            result = self._parse_reasoning_response(question, response.content, evidences)
            
            # 添加数学分析结果
            if math_analysis:
                result["math_analysis"] = math_analysis
                if "formulas" not in result:
                    result["formulas"] = math_analysis.get("formulas_found", [])
            
            return result
        except Exception as e:
            print(f"[Warning] 综合推理失败: {e}")
            return self._fallback_reasoning(question, evidences, captions)
    
    def _mixed_reasoning(self, question: str, evidences: List[Dict],
                         captions: List[Dict]) -> Dict[str, Any]:
        """混合推理（文本+数学）"""
        # 同时进行文本和数学分析
        text_result = self._text_reasoning(question, evidences, captions)
        
        # 如果有数学LLM，补充数学分析
        if self.math_llm and self.math_llm != self.text_llm:
            from ..tools.llm_client import Message
            
            evidence_context = self._build_evidence_context(evidences, captions)
            math_messages = [
                Message(role="system", content=MATH_REASONING_PROMPT),
                Message(role="user", content=f"请分析以下内容中的数学公式：\n\n{evidence_context}")
            ]
            
            try:
                math_response = self.math_llm.chat(math_messages, temperature=0.2, max_tokens=2048)
                json_match = re.search(r'\{[\s\S]*\}', math_response.content)
                if json_match:
                    math_analysis = json.loads(json_match.group())
                    text_result["math_analysis"] = math_analysis
                    if math_analysis.get("formulas_found"):
                        text_result["formulas"] = math_analysis["formulas_found"]
            except:
                pass
        
        return text_result
    
    def _parse_reasoning_response(self, question: str, response: str,
                                   evidences: List[Dict]) -> Dict[str, Any]:
        """解析推理响应"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                # 构建推理追踪
                trace = self._build_reasoning_trace(question, result, evidences)
                
                return {
                    "reasoning_trace": trace,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5),
                    "analysis_type": result.get("analysis_type", "text"),
                    "formulas": result.get("formulas", [])
                }
        except Exception as e:
            print(f"[Warning] 解析推理响应失败: {e}")
        
        return self._fallback_reasoning(question, evidences, [])
    
    def _fallback_reasoning(self, question: str, evidences: List[Dict],
                            captions: List[Dict]) -> Dict[str, Any]:
        """回退推理 - 使用简化的 LLM 调用来整理答案"""
        trace_id = secrets.token_hex(16)
        
        thinking_steps = [
            ThinkingStep(step_id=1, title="问题理解", 
                        content=f"用户询问：「{question}」"),
            ThinkingStep(step_id=2, title="证据分析",
                        content=f"检索到 {len(evidences)} 条相关证据")
        ]
        
        # 尝试使用简化的 LLM 调用来整理答案
        answer = ""
        confidence = 0.4
        
        if self.text_llm and evidences:
            try:
                from ..tools.llm_client import Message
                
                # 构建简化的证据上下文
                evidence_text = ""
                for i, ev in enumerate(evidences[:10]):
                    source = ev.get("source", "未知")
                    page = ev.get("page", "?")
                    text = ev.get("text", "")[:500]
                    evidence_text += f"[来源{i+1}: {source}, p.{page}]\n{text}\n\n"
                
                simple_prompt = f"""请根据以下证据，回答用户的问题。要求：
1. 综合整理证据内容，给出清晰的回答
2. 在关键信息后标注引用来源，如 [来源1]
3. 如果证据不足，说明局限性
4. 数学公式使用 LaTeX 格式，如 $E=mc^2$

问题：{question}

证据：
{evidence_text}

请直接给出整理后的回答："""
                
                messages = [
                    Message(role="user", content=simple_prompt)
                ]
                
                response = self.text_llm.chat(messages, temperature=0.3, max_tokens=2048)
                answer = response.content
                confidence = 0.6
                
                thinking_steps.append(ThinkingStep(
                    step_id=3, title="答案整理",
                    content="已使用 LLM 整理证据并生成回答"
                ))
                
            except Exception as e:
                print(f"[Warning] 回退 LLM 调用失败: {e}")
        
        # 如果 LLM 调用也失败，使用简单拼接
        if not answer:
            answer_parts = [f"根据检索到的证据，关于「{question}」的相关信息如下：\n"]
            for i, ev in enumerate(evidences[:5]):
                source = ev.get("source", "未知")
                page = ev.get("page", "?")
                text = ev.get("text", "")[:300]
                answer_parts.append(f"[来源{i+1}: {source}, p.{page}] {text}")
            answer = "\n\n".join(answer_parts)
            confidence = 0.3
        
        # 提取文档来源
        document_sources = []
        seen_docs = set()
        for ev in evidences:
            doc_id = ev.get("doc_id", "")
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                document_sources.append({
                    "doc_id": doc_id,
                    "source": ev.get("source", ""),
                    "pages": [ev.get("page", 0)]
                })
        
        trace = ReasoningTrace(
            trace_id=trace_id,
            question=question,
            thinking_steps=thinking_steps,
            document_sources=document_sources,
            final_answer=answer,
            confidence=confidence
        )
        
        return {
            "reasoning_trace": trace,
            "answer": answer,
            "confidence": confidence,
            "analysis_type": "text"
        }
    
    def _build_evidence_context(self, evidences: List[Dict], 
                                 captions: List[Dict]) -> str:
        """构建证据上下文"""
        context_parts = []
        
        for i, ev in enumerate(evidences[:15]):  # 增加到15条
            source = ev.get("source", "未知来源")
            page = ev.get("page", "?")
            text = ev.get("text", "")[:600]  # 增加文本长度
            
            context_parts.append(
                f"[证据{i+1}] 来源: {source} | 页码: {page}\n{text}"
            )
        
        for i, cap in enumerate(captions[:5]):
            caption_text = cap.get("generated_caption", cap.get("caption", ""))
            if caption_text:
                context_parts.append(f"[图像{i+1}] {caption_text[:300]}")
        
        return "\n\n".join(context_parts)
    
    def _build_reasoning_trace(self, question: str, result: Dict,
                                evidences: List[Dict]) -> ReasoningTrace:
        """构建推理追踪"""
        trace_id = secrets.token_hex(16)
        
        thinking_steps = []
        for step_data in result.get("thinking_steps", []):
            thinking_steps.append(ThinkingStep(
                step_id=step_data.get("step_id", 0),
                title=step_data.get("title", ""),
                content=step_data.get("content", ""),
                sources=step_data.get("sources", [])
            ))
        
        # 提取文档来源
        document_sources = []
        seen_docs = set()
        for ev in evidences:
            doc_id = ev.get("doc_id", "")
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                document_sources.append({
                    "doc_id": doc_id,
                    "source": ev.get("source", ""),
                    "pages": [ev.get("page", 0)]
                })
        
        return ReasoningTrace(
            trace_id=trace_id,
            question=question,
            thinking_steps=thinking_steps,
            document_sources=document_sources,
            final_answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.5),
            calibration_details=result.get("calibration", {})
        )
    
    def _create_empty_trace(self, question: str) -> ReasoningTrace:
        """创建空的推理追踪"""
        return ReasoningTrace(
            trace_id=secrets.token_hex(16),
            question=question,
            thinking_steps=[
                ThinkingStep(step_id=1, title="问题理解", 
                            content=f"用户询问：「{question}」"),
                ThinkingStep(step_id=2, title="证据检索", 
                            content="未找到相关证据")
            ],
            document_sources=[],
            final_answer="",
            confidence=0.0
        )
    
    def reason_stream(self, question: str, evidences: List[Dict],
                      captions: List[Dict] = None) -> Generator[Dict, None, None]:
        """流式输出推理过程"""
        import time
        captions = captions or []
        start_time = time.time()
        
        yield {"type": "thinking_start", "message": "开始多模型协同推理..."}
        
        # 分析问题类型
        analysis_type = self._analyze_question_type(question, evidences)
        yield {"type": "thinking_delta", 
               "content": f"问题类型分析：{analysis_type}（{'数学推理' if analysis_type == 'math' else '混合推理' if analysis_type == 'mixed' else '文本推理'}）\n\n"}
        
        # 证据分析
        yield {"type": "thinking_delta", 
               "content": f"正在分析 {len(evidences)} 条证据...\n"}
        
        for i, ev in enumerate(evidences[:5]):
            source = ev.get("source", "未知")
            score = ev.get("score", 0)
            yield {"type": "thinking_delta",
                   "content": f"• 证据{i+1}: {source} (相关度: {score:.2f})\n"}
        
        # 执行推理
        yield {"type": "thinking_delta", "content": "\n正在进行深度推理分析...\n"}
        
        result = self.run(AgentContext(
            question=question,
            evidences=evidences,
            captions=captions
        ))
        
        if result.success:
            data = result.data
            
            # 输出推理步骤
            trace = data.get("reasoning_trace")
            if trace and hasattr(trace, "thinking_steps"):
                for step in trace.thinking_steps:
                    yield {"type": "thinking_delta",
                           "content": f"\n【{step.title}】\n{step.content}\n"}
            
            # 如果有公式分析
            if data.get("formulas"):
                yield {"type": "thinking_delta", "content": "\n【公式分析】\n"}
                for formula in data["formulas"]:
                    yield {"type": "thinking_delta",
                           "content": f"• {formula.get('latex', '')}: {formula.get('explanation', '')}\n"}
            
            elapsed_time = time.time() - start_time
            yield {"type": "thinking_complete", "elapsed_seconds": round(elapsed_time, 1)}
            
            yield {
                "type": "final_result",
                "reasoning_trace": trace.to_dict() if hasattr(trace, "to_dict") else trace,
                "answer": data.get("answer", ""),
                "confidence": data.get("confidence", 0.5),
                "analysis_type": data.get("analysis_type", "text"),
                "formulas": data.get("formulas", [])
            }
    
    # ============================================================
    # 兼容原 ReasonerAgent 的接口
    # ============================================================
    
    def reason(self, 
               question: str, 
               evidences: List[Dict[str, Any]], 
               captions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        兼容原 ReasonerAgent 的推理方法
        
        Args:
            question: 问题
            evidences: 证据列表
            captions: 图像描述列表
            
        Returns:
            包含answer和citations的字典
        """
        context = AgentContext(
            question=question,
            evidences=evidences,
            captions=captions or []
        )
        result = self.run(context)
        
        if result.success:
            data = result.data
            # 转换为原 ReasonerAgent 的输出格式
            return {
                "answer": data.get("answer", ""),
                "key_points": data.get("key_points", []),
                "citations": data.get("citations", []),
                "confidence_factors": data.get("confidence_factors", {
                    "evidence_quality": data.get("confidence", 0.5),
                    "evidence_coverage": min(len(evidences) / 5, 1.0) if evidences else 0,
                    "reasoning_clarity": 0.7
                }),
                "limitations": data.get("limitations", "")
            }
        return {"answer": "", "citations": []}
    
    def _reason_simple(self,
                       question: str,
                       evidences: List[Dict[str, Any]],
                       captions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """简单的证据合成（兼容原 ReasonerAgent）"""
        captions = captions or []
        
        # 收集文本
        texts = []
        citations = []
        
        for i, ev in enumerate(evidences[:10]):
            text = ev.get("text", "").strip()
            if text:
                texts.append(f"[来源{i+1}] {text[:300]}")
                citations.append({
                    "id": i + 1,
                    "source": ev.get("source", ""),
                    "doc_id": ev.get("doc_id", ""),
                    "page": ev.get("page", 0),
                    "quote": text[:100],
                    "score": ev.get("score", 0)
                })
        
        # 添加图像描述
        for cap in captions[:5]:
            caption_text = cap.get("generated_caption", cap.get("caption", ""))
            if caption_text:
                texts.append(f"[图像] {caption_text[:200]}")
        
        # 合成答案
        if texts:
            answer = f"根据检索到的文献，关于「{question}」的相关信息如下：\n\n"
            answer += "\n\n".join(texts[:5])
        else:
            answer = "未找到足够的证据来回答该问题。"
        
        # 计算置信度因子
        scores = [ev.get("score", 0) for ev in evidences]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "answer": answer,
            "key_points": [],
            "citations": citations,
            "confidence_factors": {
                "evidence_quality": avg_score,
                "evidence_coverage": min(len(evidences) / 5, 1.0),
                "reasoning_clarity": 0.5  # 简单合成的清晰度较低
            },
            "limitations": "此答案由简单证据合成生成，未经深度推理。",
            "confidence": avg_score * 0.8
        }


# ============================================================
# 兼容别名（保持向后兼容）
# ============================================================

# 为了兼容原来使用 ReasonerAgent 的代码
ReasonerAgent = MultiLLMReasonerAgent
ChainOfThoughtReasoner = MultiLLMReasonerAgent
