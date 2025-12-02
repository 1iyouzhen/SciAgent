"""
Image Generator Agent - 基于论文内容生成可视化图片
支持实际图片生成（使用 Stable Diffusion 等模型）
"""
import re
import json
import secrets
import base64
import requests
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv

# 尝试加载 .env 文件（override=True 强制覆盖已存在的环境变量）
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[Info] 已加载 .env 文件: {env_path}")

from .base import BaseAgent, AgentContext, AgentResult


IMAGE_PROMPT_GENERATION = """你是一个专业的科学可视化专家。请根据用户的需求和论文内容，生成一个适合AI图片生成模型的英文提示词。

要求：
1. 提示词必须是英文
2. 描述要具体、清晰，适合生成科学示意图
3. 包含风格描述（如：scientific diagram, flowchart, infographic等）
4. 内容必须与论文相关

输出格式（JSON）：
{
    "image_prompt": "英文图片生成提示词，描述要生成的图片内容和风格",
    "negative_prompt": "不希望出现的元素（英文）",
    "style": "diagram|flowchart|infographic|illustration|chart",
    "title_zh": "中文标题",
    "description_zh": "中文描述，解释这张图片展示的内容"
}

示例输出：
{
    "image_prompt": "A scientific flowchart showing the transformer architecture with attention mechanism, clean minimalist style, blue and white color scheme, professional diagram, labeled components",
    "negative_prompt": "blurry, low quality, text, watermark, photo, realistic",
    "style": "flowchart",
    "title_zh": "Transformer架构流程图",
    "description_zh": "展示Transformer模型的核心架构，包括注意力机制的工作流程"
}"""


class ImageGeneratorAgent(BaseAgent):
    """
    图片生成Agent - 基于论文内容生成实际图片
    
    功能：
    - 分析论文内容，生成图片提示词
    - 调用图片生成API生成实际图片
    - 支持多种图片生成服务（硅基流动、Stability AI等）
    """
    
    def __init__(self,
                 config: Dict[str, Any] = None,
                 llm_client=None,
                 vl_client=None):
        super().__init__(name="image_generator", config=config)
        self.llm_client = llm_client
        self.vl_client = vl_client
        
        # 图片生成配置 - 优先从 config 读取，其次从环境变量读取
        config = config or {}
        self.image_api_provider = config.get("image_api_provider") or os.environ.get("IMAGE_API_PROVIDER", "siliconflow")
        
        # API Key 优先级: config > 环境变量
        self.image_api_key = config.get("image_api_key") or os.environ.get("SILICONFLOW_API_KEY", "")
        
        # 图片生成模型 (默认使用免费的 FLUX.1-schnell)
        self.image_model = config.get("image_model") or os.environ.get("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
        
        # 图片生成参数
        self.image_size = config.get("image_size") or os.environ.get("IMAGE_SIZE", "1024x1024")
        self.num_inference_steps = int(config.get("num_inference_steps") or os.environ.get("IMAGE_STEPS", "30"))
        self.guidance_scale = float(config.get("guidance_scale") or os.environ.get("IMAGE_GUIDANCE", "7.5"))
        
        # 输出目录
        self.output_dir = Path(config.get("output_dir") or os.environ.get("IMAGE_OUTPUT_DIR", "sci_agent/data/generated_images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打印配置信息
        if self.image_api_key:
            print(f"[Info] 图片生成配置: provider={self.image_api_provider}, model={self.image_model}")
        else:
            print("[Warning] 未配置图片生成API密钥，请在 .env 中设置 SILICONFLOW_API_KEY")
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行图片生成"""
        try:
            user_prompt = context.question
            evidences = context.evidences
            
            if not evidences:
                return AgentResult(
                    success=False,
                    error="没有可用的文档内容"
                )
            
            # 生成图片提示词
            print(f"[Info] 正在生成图片提示词...")
            prompt_result = self._generate_image_prompt(user_prompt, evidences)
            
            if not prompt_result.get("image_prompt"):
                return AgentResult(
                    success=False,
                    error="无法生成图片提示词"
                )
            
            print(f"[Info] 图片提示词: {prompt_result.get('image_prompt', '')[:100]}...")
            
            # 调用图片生成API
            print(f"[Info] 正在调用图片生成API...")
            image_result = self._generate_image(prompt_result)
            
            # 检查图片生成是否成功
            if not image_result.get("success"):
                return AgentResult(
                    success=False,
                    error=image_result.get("error", "图片生成失败")
                )
            
            return AgentResult(success=True, data={
                "prompt_info": prompt_result,
                "image_result": image_result,
                "visualization": {
                    "title": prompt_result.get("title_zh", "生成的图片"),
                    "description": prompt_result.get("description_zh", ""),
                    "style": prompt_result.get("style", "diagram"),
                    "image_url": image_result.get("image_url"),
                    "image_base64": image_result.get("image_base64"),
                    "image_path": image_result.get("image_path"),
                    "generated_at": datetime.now().isoformat()
                }
            })
        except Exception as e:
            import traceback
            print(f"[Error] 图片生成异常: {e}")
            traceback.print_exc()
            return AgentResult(success=False, error=f"图片生成异常: {str(e)}")
    
    def _generate_image_prompt(self, user_prompt: str, 
                                evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成图片提示词"""
        if not self.llm_client:
            return self._simple_prompt(user_prompt, evidences)
        
        from ..tools.llm_client import Message
        
        # 构建文档内容摘要
        doc_summary = self._build_document_summary(evidences)
        
        messages = [
            Message(role="system", content=IMAGE_PROMPT_GENERATION),
            Message(role="user", content=f"""用户需求：{user_prompt}

论文内容摘要：
{doc_summary}

请生成适合AI图片生成的提示词。""")
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.7, max_tokens=1024)
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            print(f"[Warning] 生成图片提示词失败: {e}")
        
        return self._simple_prompt(user_prompt, evidences)
    
    def _simple_prompt(self, user_prompt: str, 
                        evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简单的提示词生成"""
        # 提取关键词
        all_text = " ".join([ev.get("text", "")[:200] for ev in evidences[:5]])
        
        return {
            "image_prompt": f"Scientific diagram illustrating {user_prompt}, clean professional style, labeled components, blue color scheme",
            "negative_prompt": "blurry, low quality, text, watermark, photo",
            "style": "diagram",
            "title_zh": f"关于「{user_prompt[:20]}」的示意图",
            "description_zh": "基于论文内容自动生成的科学示意图"
        }
    
    def _build_document_summary(self, evidences: List[Dict[str, Any]]) -> str:
        """构建文档摘要"""
        summary_parts = []
        for i, ev in enumerate(evidences[:10]):
            text = ev.get("text", "")[:300]
            source = ev.get("source", "")
            if text:
                summary_parts.append(f"[{source}] {text}")
        return "\n".join(summary_parts)[:2000]
    
    def _generate_image(self, prompt_info: Dict[str, Any]) -> Dict[str, Any]:
        """调用图片生成API"""
        image_prompt = prompt_info.get("image_prompt", "")
        negative_prompt = prompt_info.get("negative_prompt", "")
        
        if not image_prompt:
            return {"success": False, "error": "缺少图片提示词"}
        
        if not self.image_api_key:
            return {
                "success": False, 
                "error": "未配置图片生成API密钥，请设置 SILICONFLOW_API_KEY 环境变量"
            }
        
        try:
            if self.image_api_provider == "siliconflow":
                return self._generate_with_siliconflow(image_prompt, negative_prompt)
            else:
                return {"success": False, "error": f"不支持的图片生成服务: {self.image_api_provider}"}
        except Exception as e:
            return {"success": False, "error": f"图片生成失败: {str(e)}"}
    
    def _generate_with_siliconflow(self, prompt: str, negative_prompt: str) -> Dict[str, Any]:
        """使用硅基流动API生成图片"""
        url = "https://api.siliconflow.cn/v1/images/generations"
        
        headers = {
            "Authorization": f"Bearer {self.image_api_key}",
            "Content-Type": "application/json"
        }
        
        # 基础参数
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "image_size": self.image_size,
        }
        
        # FLUX 模型使用不同的参数
        if "FLUX" in self.image_model.upper():
            # FLUX 模型参数
            payload["num_inference_steps"] = min(self.num_inference_steps, 20)  # FLUX schnell 最多20步
        else:
            # Stable Diffusion 模型参数
            payload["negative_prompt"] = negative_prompt
            payload["batch_size"] = 1
            payload["num_inference_steps"] = self.num_inference_steps
            payload["guidance_scale"] = self.guidance_scale
        
        print(f"[Info] 调用图片生成API: model={self.image_model}, size={self.image_size}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except:
                pass
            return {"success": False, "error": f"API错误: {error_msg}"}
        
        result = response.json()
        
        # 处理返回的图片
        images = result.get("images", []) or result.get("data", [])
        if not images:
            return {"success": False, "error": "未返回图片数据"}
        
        image_data = images[0]
        
        # 获取图片URL或base64
        image_url = image_data.get("url", "")
        image_b64 = image_data.get("b64_json", "")
        
        # 如果有URL，下载并保存
        image_path = None
        if image_url:
            try:
                img_response = requests.get(image_url, timeout=60)
                if img_response.status_code == 200:
                    # 保存图片
                    image_id = secrets.token_hex(8)
                    image_path = self.output_dir / f"generated_{image_id}.png"
                    with open(image_path, "wb") as f:
                        f.write(img_response.content)
                    image_b64 = base64.b64encode(img_response.content).decode()
            except Exception as e:
                print(f"[Warning] 下载图片失败: {e}")
        elif image_b64:
            # 保存base64图片
            try:
                image_id = secrets.token_hex(8)
                image_path = self.output_dir / f"generated_{image_id}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_b64))
            except Exception as e:
                print(f"[Warning] 保存图片失败: {e}")
        
        return {
            "success": True,
            "image_url": image_url,
            "image_base64": image_b64,
            "image_path": str(image_path) if image_path else None,
            "prompt_used": prompt
        }
    
    def generate_visualization_from_image(self, image_path: str,
                                           user_prompt: str,
                                           evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于上传的图片和文档生成新图片"""
        # 如果有VL模型，先分析上传的图片
        image_description = ""
        if self.vl_client:
            try:
                image_description = self.vl_client.describe_image(
                    image_path, 
                    "请描述这张图片的内容，特别是其中的科学概念和结构。"
                )
            except Exception as e:
                print(f"[Warning] 图片分析失败: {e}")
        
        # 生成新的提示词，参考上传的图片
        enhanced_prompt = user_prompt
        if image_description:
            enhanced_prompt = f"{user_prompt}\n\n参考图片描述：{image_description[:500]}"
        
        # 生成图片
        prompt_result = self._generate_image_prompt(enhanced_prompt, evidences)
        image_result = self._generate_image(prompt_result)
        
        return {
            "prompt_info": prompt_result,
            "image_result": image_result,
            "source_image_description": image_description,
            "visualization": {
                "title": prompt_result.get("title_zh", "生成的图片"),
                "description": prompt_result.get("description_zh", ""),
                "style": prompt_result.get("style", "diagram"),
                "image_url": image_result.get("image_url"),
                "image_base64": image_result.get("image_base64"),
                "image_path": image_result.get("image_path"),
                "generated_at": datetime.now().isoformat()
            }
        }
