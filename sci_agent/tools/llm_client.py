"""
LLM客户端 - 统一的模型调用接口
支持多平台 API：
- dashscope: 阿里云 DashScope (Qwen系列)
- openai: OpenAI API (GPT系列)
- deepseek: DeepSeek API
- zhipu: 智谱 API (GLM系列)
- moonshot: Moonshot/Kimi API
- baichuan: 百川 API
- siliconflow: 硅基流动 API (多模型)
- ollama: 本地 Ollama
- local: 本地 Transformers 模型
"""
import os
import json
import requests
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass


class UnifiedLLM:
    """统一的 LLM API 调用类，支持多平台"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model.lower()
        self.provider = self._detect_provider()
        self.base_url = self._get_base_url()
    
    def _detect_provider(self) -> str:
        """自动识别模型属于哪家平台"""
        m = self.model
        if any(x in m for x in ["gpt-4", "gpt-3", "o1", "o3", "omni"]):
            return "openai"
        if "deepseek" in m:
            return "deepseek"
        if "glm" in m or "zhipu" in m:
            return "zhipu"
        if any(x in m for x in ["moonshot", "kamiya"]):
            return "moonshot"
        if "silicon" in m or "deepseek-ai" in m:
            return "siliconflow"
        if "qwen" in m:
            return "dashscope"
        if "baichuan" in m:
            return "baichuan"
        return "unknown"
    
    def _get_base_url(self) -> Optional[str]:
        """不同平台对应的 base_url"""
        urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "siliconflow": "https://api.siliconflow.cn/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "moonshot": "https://api.moonshot.cn/v1",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "baichuan": "https://api.baichuan-ai.com/v1",
        }
        return urls.get(self.provider)
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
        """统一调用 chat 接口"""
        if self.provider == "siliconflow":
            return self._siliconflow_chat(messages, temperature, max_tokens)
        return self._standard_chat(messages, temperature, max_tokens)
    
    def _standard_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
        """标准 OpenAI 兼容格式调用"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": {"message": "API请求超时（120秒）"}}
        except requests.exceptions.HTTPError as e:
            return {"error": {"message": f"HTTP错误: {e.response.status_code} - {e.response.text}"}}
        except requests.exceptions.RequestException as e:
            return {"error": {"message": f"请求失败: {str(e)}"}}
        except json.JSONDecodeError as e:
            return {"error": {"message": f"JSON解析失败: {str(e)}"}}
    
    def _siliconflow_chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
        """SiliconFlow 调用 - 使用标准 chat/completions 接口（支持多模态）"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": {"message": "API请求超时（120秒）"}}
        except requests.exceptions.HTTPError as e:
            return {"error": {"message": f"HTTP错误: {e.response.status_code} - {e.response.text}"}}
        except requests.exceptions.RequestException as e:
            return {"error": {"message": f"请求失败: {str(e)}"}}
        except json.JSONDecodeError as e:
            return {"error": {"message": f"JSON解析失败: {str(e)}"}}


# API 配置映射
API_CONFIGS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat"
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
        "default_model": "glm-4-flash"
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k"
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "env_key": "BAICHUAN_API_KEY",
        "default_model": "Baichuan4"
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        "default_model": "Qwen/Qwen2.5-7B-Instruct"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "qwen2.5:7b"
    }
}


@dataclass
class Message:
    """消息对象"""
    role: str  # system, user, assistant
    content: str
    images: List[str] = None  # 图像路径列表（用于VL模型）
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    usage: Dict[str, int] = None
    finish_reason: str = "stop"
    
    
class LLMClient:
    """
    统一的LLM客户端
    支持多平台 API 调用
    """
    
    def __init__(self,
                 model: str = None,
                 api_provider: str = "dashscope",
                 api_base: str = None,
                 api_key: str = None,
                 use_local: bool = False,
                 device: str = None):
        self.api_provider = api_provider.lower()
        self.use_local = use_local or self.api_provider == "local"
        self.device = device
        
        # 获取 API 配置
        config = API_CONFIGS.get(self.api_provider, {})
        
        # 设置 API base URL（优先使用传入的，其次使用 provider 对应的默认地址）
        self.api_base = api_base or config.get("base_url")
        
        # 设置 API key（优先使用传入的，其次环境变量）
        env_key = config.get("env_key", "OPENAI_API_KEY")
        self.api_key = api_key or os.environ.get(env_key, "")
        
        # 设置模型（优先使用传入的，其次默认模型）
        self.model = model or config.get("default_model", "gpt-4o-mini")
        
        self._local_model = None
        self._local_tokenizer = None
        
        # 打印配置信息（调试用）
        if not self.use_local:
            print(f"[Info] LLM配置: provider={self.api_provider}, model={self.model}")
    
    def chat(self,
             messages: List[Message],
             temperature: float = 0.7,
             max_tokens: int = 2048,
             stream: bool = False) -> LLMResponse:
        """
        聊天接口
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成长度
            stream: 是否流式输出
            
        Returns:
            LLM响应
        """
        if self.use_local:
            return self._chat_local(messages, temperature, max_tokens)
        else:
            return self._chat_api(messages, temperature, max_tokens, stream)

    def _chat_api(self,
                  messages: List[Message],
                  temperature: float,
                  max_tokens: int,
                  stream: bool) -> LLMResponse:
        """通过 UnifiedLLM 调用 API（支持所有平台）"""
        try:
            # Ollama 不需要真实 key，其他平台需要
            api_key = self.api_key if self.api_key else ("ollama" if self.api_provider == "ollama" else "")
            
            if not api_key and self.api_provider != "ollama":
                raise ValueError(f"未设置 API Key，请设置环境变量 {API_CONFIGS.get(self.api_provider, {}).get('env_key', 'API_KEY')}")
            
            # 使用 UnifiedLLM 进行调用
            unified_client = UnifiedLLM(api_key=api_key, model=self.model)
            
            # 如果有自定义 base_url，覆盖自动检测的
            if self.api_base and self.api_base != unified_client.base_url:
                unified_client.base_url = self.api_base
            
            # 转换消息格式
            api_messages = self._convert_messages(messages)
            
            # 打印调试信息
            print(f"[Debug] LLM API调用: provider={self.api_provider}, model={self.model}, base_url={unified_client.base_url}")
            print(f"[Debug] 消息数量: {len(api_messages)}, temperature={temperature}, max_tokens={max_tokens}")
            
            # 调用 API
            response = unified_client.chat(api_messages, temperature, max_tokens)
            
            # 打印原始响应（调试用）
            print(f"[Debug] API原始响应类型: {type(response)}")
            if isinstance(response, dict):
                print(f"[Debug] API响应keys: {response.keys()}")
            
            # 解析响应
            if "error" in response:
                error_msg = response.get("error", {})
                if isinstance(error_msg, dict):
                    error_detail = error_msg.get("message", str(error_msg))
                else:
                    error_detail = str(error_msg)
                print(f"[Error] API返回错误: {error_detail}")
                raise ValueError(f"API错误: {error_detail}")
            
            # 标准 OpenAI 格式响应
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                content = choice.get("message", {}).get("content", "")
                
                # 检查内容是否为空
                if not content:
                    print(f"[Warning] API返回空内容，完整响应: {response}")
                else:
                    print(f"[Debug] API调用成功，响应长度: {len(content)}")
                
                usage = response.get("usage")
                return LLMResponse(
                    content=content,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    } if usage else None,
                    finish_reason=choice.get("finish_reason", "stop")
                )
            
            # SiliconFlow responses 格式
            if "output" in response:
                output = response.get("output", "")
                print(f"[Debug] SiliconFlow格式响应，output长度: {len(output)}")
                return LLMResponse(content=output)
            
            # 其他格式尝试提取
            print(f"[Warning] 未知响应格式，尝试转换为字符串: {response}")
            return LLMResponse(content=str(response))
            
        except Exception as e:
            import traceback
            print(f"[Error] API调用失败 ({self.api_provider}): {e}")
            traceback.print_exc()
            return LLMResponse(content=f"[API调用失败: {e}]")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """转换消息格式"""
        api_messages = []
        for msg in messages:
            if msg.images:
                # 多模态消息
                content = [{"type": "text", "text": msg.content}]
                for img_path in msg.images:
                    if os.path.exists(img_path):
                        import base64
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                        })
                api_messages.append({"role": msg.role, "content": content})
            else:
                api_messages.append(msg.to_dict())
        return api_messages
    
    def _chat_local(self,
                    messages: List[Message],
                    temperature: float,
                    max_tokens: int) -> LLMResponse:
        """本地模型调用"""
        self._load_local_model()
        
        if self._local_model is None:
            return LLMResponse(content="[模型加载失败]")
        
        import torch
        
        # 构建prompt
        prompt = self._build_prompt(messages)
        
        inputs = self._local_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._local_tokenizer.eos_token_id
            )
        
        response = self._local_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return LLMResponse(content=response)
    
    def _load_local_model(self):
        """加载本地模型"""
        if self._local_model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self._local_model = self._local_model.to(self.device)
        except Exception as e:
            print(f"[Error] 无法加载本地模型: {e}")
            self._local_model = None
    
    def _build_prompt(self, messages: List[Message]) -> str:
        """构建prompt"""
        # Qwen格式
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt


class VLClient(LLMClient):
    """
    视觉语言模型客户端
    专门用于图像理解任务
    
    注意：不同平台的VL模型支持情况：
    - dashscope: qwen-vl-max, qwen-vl-plus
    - zhipu: glm-4v, glm-4v-flash
    - siliconflow: Qwen/Qwen2-VL-7B-Instruct (需要使用 chat/completions 接口)
    - openai: gpt-4o, gpt-4-vision-preview
    """
    
    def __init__(self,
                 model: str = "qwen-vl-max",
                 api_provider: str = "dashscope",
                 **kwargs):
        super().__init__(model=model, api_provider=api_provider, **kwargs)
        print(f"[Info] VLClient初始化: provider={api_provider}, model={model}")
    
    def describe_image(self, 
                       image_path: str, 
                       prompt: str = "请详细描述这张图片的内容。") -> str:
        """
        描述图像内容
        
        Args:
            image_path: 图像路径
            prompt: 提示词
            
        Returns:
            图像描述
        """
        messages = [
            Message(role="user", content=prompt, images=[image_path])
        ]
        response = self.chat(messages)
        return response.content
    
    def extract_text_from_image(self, image_path: str) -> str:
        """从图像中提取文本（OCR）"""
        prompt = "请提取并输出图片中的所有文字内容，保持原有格式。"
        return self.describe_image(image_path, prompt)
    
    def analyze_chart(self, image_path: str) -> Dict[str, Any]:
        """分析图表"""
        prompt = """请分析这张图表，输出JSON格式：
{
    "chart_type": "图表类型",
    "title": "标题",
    "x_axis": "X轴含义",
    "y_axis": "Y轴含义",
    "key_findings": ["关键发现1", "关键发现2"],
    "data_summary": "数据摘要"
}"""
        response = self.describe_image(image_path, prompt)
        
        try:
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"description": response}
