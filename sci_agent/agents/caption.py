"""
Caption Agent - 图像理解
职责：使用Qwen-VL对文档中的图像进行理解和描述
"""
import os
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentContext, AgentResult


CAPTION_SYSTEM_PROMPT = """你是一个专业的科学图像分析专家。请详细分析图像内容，包括：
1. 图像类型（图表、示意图、照片等）
2. 主要内容和关键信息
3. 数据趋势或关键发现（如果是图表）
4. 与科学研究的相关性

请用简洁专业的语言描述。"""


class CaptionAgent(BaseAgent):
    """
    Caption Agent - 图像理解
    
    功能：
    - 使用Qwen-VL进行图像理解
    - 支持图表分析
    - 支持OCR文字提取
    - 生成结构化的图像描述
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 vl_client = None):
        super().__init__(name="caption", config=config)
        self.vl_client = vl_client
    
    def run(self, context: AgentContext) -> AgentResult:
        """
        执行图像理解
        
        Args:
            context: Agent上下文
            
        Returns:
            包含图像描述的结果
        """
        evidences = context.evidences
        
        # 收集需要处理的图像
        images_to_process = []
        for ev in evidences:
            # 检查是否为图像类型的证据
            chunk_type = ev.get("chunk_type", "")
            metadata = ev.get("metadata", {})
            
            # 支持多种图像路径字段名
            image_path = (
                metadata.get("image_path", "") or 
                metadata.get("path", "") or
                ev.get("image_path", "")  # 直接在证据中的字段
            )
            
            if chunk_type == "image" or image_path:
                if image_path and os.path.exists(image_path):
                    images_to_process.append({
                        "path": image_path,
                        "doc_id": ev.get("doc_id"),
                        "page": ev.get("page"),
                        "original_caption": ev.get("text", "")
                    })
                elif chunk_type == "image":
                    # 记录找不到图像文件的情况
                    print(f"[Warning] 图像文件不存在: {image_path}")
        
        # 如果没有VL客户端，提示用户
        if not self.vl_client and images_to_process:
            print(f"[Warning] VL客户端未初始化，无法处理 {len(images_to_process)} 张图像")
        
        # 处理图像
        captions = []
        for img_info in images_to_process:
            caption = self._process_image(img_info)
            captions.append(caption)
        
        # 输出调试信息
        if not images_to_process:
            image_count = sum(1 for ev in evidences if ev.get("chunk_type") == "image")
            if image_count > 0:
                print(f"[Info] 发现 {image_count} 条图像证据，但图像文件不存在")
            else:
                print(f"[Info] 检索到的 {len(evidences)} 条证据中没有图像类型")
        
        return AgentResult(
            success=True,
            data={"captions": captions}
        )
    
    def _process_image(self, img_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个图像"""
        image_path = img_info["path"]
        
        result = {
            "path": image_path,
            "doc_id": img_info.get("doc_id"),
            "page": img_info.get("page"),
            "original_caption": img_info.get("original_caption", ""),
            "generated_caption": "",
            "image_type": "unknown",
            "key_findings": [],
            "ocr_text": ""
        }
        
        if not self.vl_client:
            # 无VL模型时使用原始caption
            result["generated_caption"] = img_info.get("original_caption", f"图像 {image_path}")
            return result
        
        try:
            # 1. 生成图像描述
            description = self.vl_client.describe_image(
                image_path, 
                prompt=CAPTION_SYSTEM_PROMPT
            )
            result["generated_caption"] = description
            
            # 2. 识别图像类型
            result["image_type"] = self._identify_image_type(description)
            
            # 3. 如果是图表，进行详细分析
            if result["image_type"] in ["chart", "graph", "table"]:
                chart_analysis = self.vl_client.analyze_chart(image_path)
                result["chart_analysis"] = chart_analysis
                result["key_findings"] = chart_analysis.get("key_findings", [])
            
            # 4. 提取OCR文字
            ocr_text = self.vl_client.extract_text_from_image(image_path)
            result["ocr_text"] = ocr_text
            
        except Exception as e:
            print(f"[Warning] 图像处理失败 {image_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def _identify_image_type(self, description: str) -> str:
        """识别图像类型"""
        description_lower = description.lower()
        
        # 图表类型关键词
        chart_keywords = ["chart", "graph", "plot", "图表", "曲线", "柱状图", "饼图", "折线图"]
        table_keywords = ["table", "表格", "数据表"]
        diagram_keywords = ["diagram", "示意图", "流程图", "结构图", "架构图"]
        photo_keywords = ["photo", "photograph", "照片", "图片", "实验"]
        
        if any(kw in description_lower for kw in chart_keywords):
            return "chart"
        elif any(kw in description_lower for kw in table_keywords):
            return "table"
        elif any(kw in description_lower for kw in diagram_keywords):
            return "diagram"
        elif any(kw in description_lower for kw in photo_keywords):
            return "photo"
        
        return "unknown"
    
    # 兼容旧接口
    def caption(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        兼容旧接口的图像描述方法
        
        Args:
            pages: 页面列表，每个包含images字段
            
        Returns:
            图像描述列表
        """
        captions = []
        
        for page in pages:
            page_num = page.get("page", 0)
            images = page.get("images", [])
            
            for i, img in enumerate(images):
                img_path = img.get("path", "") if isinstance(img, dict) else ""
                
                caption_result = {
                    "page": page_num,
                    "index": i,
                    "path": img_path,
                    "caption": ""
                }
                
                if img_path and os.path.exists(img_path) and self.vl_client:
                    try:
                        caption_result["caption"] = self.vl_client.describe_image(img_path)
                    except:
                        caption_result["caption"] = f"图像{page_num}-{i}"
                else:
                    caption_result["caption"] = f"图像{page_num}-{i}"
                
                captions.append(caption_result)
        
        return captions


class BatchCaptionAgent(CaptionAgent):
    """
    批量图像处理Agent - 支持并行处理
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 vl_client = None,
                 max_workers: int = 4):
        super().__init__(config=config, vl_client=vl_client)
        self.max_workers = max_workers
    
    def run(self, context: AgentContext) -> AgentResult:
        """批量处理图像"""
        evidences = context.evidences
        
        # 收集图像
        images_to_process = []
        for ev in evidences:
            if ev.get("chunk_type") == "image":
                image_path = ev.get("metadata", {}).get("image_path", "")
                if image_path and os.path.exists(image_path):
                    images_to_process.append({
                        "path": image_path,
                        "doc_id": ev.get("doc_id"),
                        "page": ev.get("page"),
                        "original_caption": ev.get("text", "")
                    })
        
        if not images_to_process:
            return AgentResult(success=True, data={"captions": []})
        
        # 并行处理
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            captions = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_image, img): img 
                    for img in images_to_process
                }
                
                for future in as_completed(futures):
                    try:
                        caption = future.result()
                        captions.append(caption)
                    except Exception as e:
                        print(f"[Warning] 图像处理失败: {e}")
            
            return AgentResult(success=True, data={"captions": captions})
        except ImportError:
            # 回退到串行处理
            return super().run(context)
