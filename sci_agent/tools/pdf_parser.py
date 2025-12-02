"""
PDF解析工具 - 使用MinerU进行多模态PDF解析
支持文本、表格、图像提取
"""
import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PageContent:
    """页面内容"""
    page_num: int
    text: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    layout_blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    source: str
    title: str = ""
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PdfParser:
    """
    PDF解析器 - 使用MinerU进行多模态解析
    支持：文本提取、表格识别、图像提取、布局分析
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._mineru_available = self._check_mineru()
    
    def _check_mineru(self) -> bool:
        """检查MinerU是否可用"""
        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe
            return True
        except ImportError:
            return False
    
    def _generate_doc_id(self, path: str) -> str:
        """生成文档唯一ID"""
        content = f"{path}_{os.path.getmtime(path)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        解析PDF文档
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            ParsedDocument: 解析后的文档对象
        """
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        doc_id = self._generate_doc_id(pdf_path)
        
        # 检查缓存
        cache_path = self.output_dir / f"{doc_id}.json"
        if cache_path.exists():
            return self._load_cache(cache_path)
        
        # 根据文件类型选择解析方式
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            if self._mineru_available:
                doc = self._parse_with_mineru(pdf_path, doc_id)
            else:
                doc = self._parse_pdf_fallback(pdf_path, doc_id)
        elif suffix == ".txt":
            doc = self._parse_txt(pdf_path, doc_id)
        else:
            doc = self._parse_generic(pdf_path, doc_id)
        
        # 保存缓存
        self._save_cache(doc, cache_path)
        return doc
    
    def _parse_with_mineru(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        """使用MinerU解析PDF"""
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
        
        p = Path(pdf_path)
        output_path = self.output_dir / doc_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        # 创建读写器
        image_writer = DiskReaderWriter(str(output_path / "images"))
        
        # 解析PDF
        pipe = UNIPipe(pdf_bytes, [], image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        
        # 获取解析结果
        md_content = pipe.pipe_mk_markdown(str(output_path / "images"))
        content_list = pipe.pipe_mk_uni_format(str(output_path / "images"))
        
        # 构建文档对象
        pages = []
        current_page = 1
        page_content = PageContent(page_num=current_page)
        
        for block in content_list:
            block_page = block.get("page_idx", 0) + 1
            if block_page != current_page:
                pages.append(page_content)
                current_page = block_page
                page_content = PageContent(page_num=current_page)
            
            block_type = block.get("type", "")
            if block_type == "text":
                page_content.text += block.get("text", "") + "\n"
            elif block_type == "table":
                page_content.tables.append({
                    "html": block.get("html", ""),
                    "text": block.get("text", "")
                })
            elif block_type == "image":
                page_content.images.append({
                    "path": block.get("img_path", ""),
                    "caption": block.get("img_caption", "")
                })
            
            page_content.layout_blocks.append(block)
        
        pages.append(page_content)
        
        return ParsedDocument(
            doc_id=doc_id,
            source=str(p),
            title=p.stem,
            pages=pages,
            metadata={"parser": "mineru", "markdown": md_content}
        )
    
    def _parse_pdf_fallback(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        """PDF解析回退方案 - 使用PyPDF2（仅支持文本提取，不支持图像）"""
        try:
            import PyPDF2
            pages = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append(PageContent(page_num=i+1, text=text))
            
            print(f"[Warning] 使用PyPDF2解析，不支持图像提取。如需图像理解功能，请安装MinerU: pip install magic-pdf")
            
            return ParsedDocument(
                doc_id=doc_id,
                source=pdf_path,
                title=Path(pdf_path).stem,
                pages=pages,
                metadata={"parser": "pypdf2", "image_support": False}
            )
        except ImportError:
            # 最简单的回退
            return ParsedDocument(
                doc_id=doc_id,
                source=pdf_path,
                title=Path(pdf_path).stem,
                pages=[PageContent(page_num=1, text="[PDF解析需要安装PyPDF2或MinerU]")],
                metadata={"parser": "none", "image_support": False}
            )
    
    def _parse_txt(self, txt_path: str, doc_id: str) -> ParsedDocument:
        """解析文本文件"""
        p = Path(txt_path)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="gbk", errors="ignore")
        
        return ParsedDocument(
            doc_id=doc_id,
            source=str(p),
            title=p.stem,
            pages=[PageContent(page_num=1, text=text)],
            metadata={"parser": "text"}
        )
    
    def _parse_generic(self, file_path: str, doc_id: str) -> ParsedDocument:
        """通用文件解析"""
        p = Path(file_path)
        return ParsedDocument(
            doc_id=doc_id,
            source=str(p),
            title=p.stem,
            pages=[PageContent(page_num=1, text=f"[不支持的文件格式: {p.suffix}]")],
            metadata={"parser": "generic"}
        )
    
    def _save_cache(self, doc: ParsedDocument, cache_path: Path) -> None:
        """保存解析缓存"""
        data = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "title": doc.title,
            "metadata": doc.metadata,
            "pages": [
                {
                    "page_num": p.page_num,
                    "text": p.text,
                    "tables": p.tables,
                    "images": p.images,
                    "layout_blocks": p.layout_blocks
                }
                for p in doc.pages
            ]
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_cache(self, cache_path: Path) -> ParsedDocument:
        """加载解析缓存"""
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        pages = [
            PageContent(
                page_num=p["page_num"],
                text=p["text"],
                tables=p.get("tables", []),
                images=p.get("images", []),
                layout_blocks=p.get("layout_blocks", [])
            )
            for p in data["pages"]
        ]
        
        return ParsedDocument(
            doc_id=data["doc_id"],
            source=data["source"],
            title=data.get("title", ""),
            pages=pages,
            metadata=data.get("metadata", {})
        )
    
    def to_chunks(self, doc: ParsedDocument, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        将文档切分为检索块
        
        Args:
            doc: 解析后的文档
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            检索块列表
        """
        chunks = []
        
        for page in doc.pages:
            text = page.text.strip()
            if not text:
                continue
            
            # 文本切分
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                
                chunks.append({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "page": page.page_num,
                    "text": chunk_text,
                    "chunk_type": "text",
                    "start_pos": start,
                    "end_pos": end
                })
                
                start = end - overlap if end < len(text) else end
            
            # 表格作为独立块
            for i, table in enumerate(page.tables):
                chunks.append({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "page": page.page_num,
                    "text": table.get("text", ""),
                    "html": table.get("html", ""),
                    "chunk_type": "table",
                    "table_index": i
                })
            
            # 图像作为独立块
            for i, img in enumerate(page.images):
                chunks.append({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "page": page.page_num,
                    "text": img.get("caption", ""),
                    "image_path": img.get("path", ""),
                    "chunk_type": "image",
                    "image_index": i
                })
        
        return chunks
