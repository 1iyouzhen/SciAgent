"""
综述报告生成Agent - 根据上传文档自动生成综述报告
每句话标注来源引用，支持引用定位
"""
import re
import json
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentContext, AgentResult


REPORT_GENERATION_PROMPT = """你是一个专业的科学文献综述撰写专家。请根据提供的文档内容，按照用户需求生成综述报告。

要求：
1. 每个关键论点必须标注引用，格式：[来源X]
2. 引用必须准确对应提供的证据
3. 报告结构清晰，逻辑连贯
4. 使用专业、准确的学术语言
5. 综合多个文档的信息，避免简单罗列
6. 数学公式必须使用LaTeX格式：
   - 行内公式用单美元符号：$E=mc^2$、$\\alpha$、$\\frac{a}{b}$
   - 块级公式用双美元符号：$$\\sum_{i=1}^n x_i$$
   - 所有数学符号、变量、方程都使用LaTeX

输出格式（JSON）：
{
    "title": "报告标题",
    "abstract": "摘要（100-200字）",
    "sections": [
        {
            "heading": "章节标题",
            "content": "章节内容，每个关键论点标注[来源X]，数学公式用LaTeX如 $x^2$",
            "citations_used": [1, 2, 3]
        }
    ],
    "conclusion": "结论部分",
    "citations": [
        {
            "id": 1,
            "doc_id": "文档ID",
            "source": "文档名称",
            "page": 页码,
            "quote": "引用原文",
            "relevance": "与报告的关联说明"
        }
    ],
    "confidence": 0.85
}"""


QUESTION_GENERATION_PROMPT = """你是一个科学研究助手。根据以下文档内容，生成3-5个有价值的研究问题。

要求：
1. 问题应该能够通过这些文档回答
2. 问题应该有深度，不是简单的事实查询
3. 问题应该涵盖文档的主要内容
4. 问题表述清晰、具体

输出格式（JSON）：
{
    "questions": [
        "问题1",
        "问题2",
        "问题3"
    ]
}"""


class ReportGeneratorAgent(BaseAgent):
    """
    综述报告生成Agent
    
    功能：
    - 根据多文档生成综述报告
    - 每句话标注来源引用
    - 支持引用定位到具体文档
    - 生成推荐问题
    """
    
    def __init__(self,
                 config: Dict[str, Any] = None,
                 llm_client = None):
        super().__init__(name="report_generator", config=config)
        self.llm_client = llm_client
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行报告生成"""
        user_requirement = context.question  # 用户的报告需求
        evidences = context.evidences
        
        if not evidences:
            return AgentResult(
                success=False,
                error="没有可用的文档证据"
            )
        
        if self.llm_client:
            result = self._generate_with_llm(user_requirement, evidences)
        else:
            result = self._generate_simple(user_requirement, evidences)
        
        return AgentResult(success=True, data=result)
    
    def _generate_with_llm(self, requirement: str, 
                           evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM生成报告"""
        from ..tools.llm_client import Message
        
        # 构建证据上下文
        evidence_context = self._build_evidence_context(evidences)
        
        user_prompt = f"""用户需求：{requirement}

文档证据：
{evidence_context}

请根据以上文档内容，按照用户需求生成综述报告。"""
        
        messages = [
            Message(role="system", content=REPORT_GENERATION_PROMPT),
            Message(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3, max_tokens=4096)
            
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                
                # 处理引用，添加文档定位信息
                citations = result.get("citations", [])
                for i, cite in enumerate(citations):
                    cite["id"] = cite.get("id", i + 1)
                    # 关联原始证据
                    if i < len(evidences):
                        cite["doc_id"] = cite.get("doc_id", evidences[i].get("doc_id", ""))
                        cite["file_path"] = evidences[i].get("file_path", "")
                
                return {
                    "report": result,
                    "citations": citations,
                    "confidence": result.get("confidence", 0.7)
                }
        except Exception as e:
            print(f"[Warning] LLM报告生成失败: {e}")
        
        return self._generate_simple(requirement, evidences)
    
    def _generate_simple(self, requirement: str,
                         evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简单的报告生成（无LLM）"""
        # 按文档分组证据
        doc_groups: Dict[str, List[Dict]] = {}
        for ev in evidences:
            doc_id = ev.get("doc_id", "unknown")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(ev)
        
        # 构建报告
        sections = []
        citations = []
        cite_id = 1
        
        for doc_id, doc_evidences in doc_groups.items():
            source = doc_evidences[0].get("source", "未知来源") if doc_evidences else "未知来源"
            
            section_content = []
            section_citations = []
            
            for ev in doc_evidences[:5]:  # 每个文档最多5条证据
                text = ev.get("text", "")[:300]
                page = ev.get("page", 0)
                
                section_content.append(f"{text} [来源{cite_id}]")
                section_citations.append(cite_id)
                
                citations.append({
                    "id": cite_id,
                    "doc_id": doc_id,
                    "source": source,
                    "page": page,
                    "quote": text[:100],
                    "file_path": ev.get("file_path", "")
                })
                cite_id += 1
            
            sections.append({
                "heading": f"来自 {Path(source).stem if source else '文档'}",
                "content": "\n\n".join(section_content),
                "citations_used": section_citations
            })
        
        report = {
            "title": f"关于「{requirement[:50]}」的综述报告",
            "abstract": f"本报告基于 {len(doc_groups)} 个文档，共 {len(evidences)} 条证据进行综合分析。",
            "sections": sections,
            "conclusion": "以上是基于提供文档的综合分析，建议结合更多文献进行深入研究。",
            "citations": citations,
            "confidence": 0.5
        }
        
        return {
            "report": report,
            "citations": citations,
            "confidence": 0.5
        }
    
    def _build_evidence_context(self, evidences: List[Dict[str, Any]]) -> str:
        """构建证据上下文"""
        context_parts = []
        
        for i, ev in enumerate(evidences[:30]):  # 限制数量
            source = ev.get("source", "未知来源")
            page = ev.get("page", "?")
            text = ev.get("text", "")[:500]
            doc_id = ev.get("doc_id", "")
            
            context_parts.append(
                f"[证据{i+1}] 文档: {source} | 页码: {page} | ID: {doc_id}\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_questions(self, evidences: List[Dict[str, Any]]) -> List[str]:
        """
        根据文档生成推荐问题
        
        Args:
            evidences: 文档证据列表
            
        Returns:
            推荐问题列表
        """
        if not evidences:
            return []
        
        if self.llm_client:
            return self._generate_questions_with_llm(evidences)
        else:
            return self._generate_questions_simple(evidences)
    
    def _generate_questions_with_llm(self, evidences: List[Dict[str, Any]]) -> List[str]:
        """使用LLM生成问题"""
        from ..tools.llm_client import Message
        
        # 构建文档摘要
        doc_summary = []
        seen_docs = set()
        
        for ev in evidences[:20]:
            doc_id = ev.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            source = ev.get("source", "未知")
            text = ev.get("text", "")[:300]
            doc_summary.append(f"文档: {source}\n内容摘要: {text}")
        
        user_prompt = f"""以下是用户上传的文档内容摘要：

{chr(10).join(doc_summary)}

请生成3-5个有价值的研究问题。"""
        
        messages = [
            Message(role="system", content=QUESTION_GENERATION_PROMPT),
            Message(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.5)
            
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("questions", [])[:5]
        except Exception as e:
            print(f"[Warning] 问题生成失败: {e}")
        
        return self._generate_questions_simple(evidences)
    
    def _generate_questions_simple(self, evidences: List[Dict[str, Any]]) -> List[str]:
        """简单的问题生成"""
        questions = []
        
        # 收集文档主题
        sources = set()
        for ev in evidences:
            source = ev.get("source", "")
            if source:
                sources.add(Path(source).stem if source else "")
        
        # 生成通用问题
        if sources:
            source_list = list(sources)[:3]
            questions.append(f"请总结{source_list[0]}的主要研究发现")
            
            if len(source_list) > 1:
                questions.append(f"比较{source_list[0]}和{source_list[1]}的研究方法有何异同？")
            
            questions.append("这些文档中提到的主要研究挑战是什么？")
            questions.append("基于这些文献，未来的研究方向可能是什么？")
            questions.append("这些研究的实际应用价值是什么？")
        
        return questions[:5]


# 需要导入Path
from pathlib import Path
