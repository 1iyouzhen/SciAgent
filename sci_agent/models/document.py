"""
文档管理模块 - 支持多文档上传、URL下载、文档缓存
"""
import os
import json
import hashlib
import secrets
import re
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urlparse
import tempfile


@dataclass
class UploadedDocument:
    """上传的文档"""
    doc_id: str
    filename: str
    file_type: str  # pdf, txt, docx, url
    file_path: str  # 本地存储路径
    original_url: str = ""  # 如果是URL来源
    file_size: int = 0
    page_count: int = 0
    chunk_count: int = 0
    status: str = "pending"  # pending, processing, ready, error
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UploadedDocument":
        return cls(**data)


@dataclass
class DocumentSession:
    """文档会话 - 一轮对话中的所有文档"""
    session_id: str
    conversation_id: str
    user_id: str
    documents: List[UploadedDocument] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "documents": [d.to_dict() for d in self.documents],
            "suggested_questions": self.suggested_questions,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSession":
        docs = [UploadedDocument.from_dict(d) for d in data.get("documents", [])]
        return cls(
            session_id=data["session_id"],
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            documents=docs,
            suggested_questions=data.get("suggested_questions", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", "")
        )


class DocumentManager:
    """文档管理器"""
    
    MAX_DOCUMENTS_PER_SESSION = 50
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".tex"}
    
    def __init__(self, data_dir: str = "data/documents"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.data_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, DocumentSession] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """加载所有会话"""
        for file_path in self.sessions_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = DocumentSession.from_dict(data)
                    self._sessions[session.session_id] = session
            except Exception as e:
                print(f"[Warning] 加载文档会话失败: {file_path}, {e}")
    
    def _save_session(self, session: DocumentSession):
        """保存会话"""
        file_path = self.sessions_dir / f"{session.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def create_session(self, conversation_id: str, user_id: str) -> DocumentSession:
        """创建文档会话"""
        session_id = secrets.token_hex(16)
        session = DocumentSession(
            session_id=session_id,
            conversation_id=conversation_id,
            user_id=user_id
        )
        self._sessions[session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[DocumentSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def get_session_by_conversation(self, conversation_id: str) -> Optional[DocumentSession]:
        """通过对话ID获取会话"""
        for session in self._sessions.values():
            if session.conversation_id == conversation_id:
                return session
        return None
    
    def get_user_sessions(self, user_id: str) -> List[DocumentSession]:
        """获取用户的所有文档会话"""
        return [s for s in self._sessions.values() if s.user_id == user_id]
    
    def add_document_from_file(self, session_id: str, filename: str, 
                               file_content: bytes) -> Optional[UploadedDocument]:
        """从文件添加文档"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if len(session.documents) >= self.MAX_DOCUMENTS_PER_SESSION:
            return None
        
        # 检查文件类型
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return None
        
        # 生成文档ID和保存路径 - 保留原文件名
        doc_id = secrets.token_hex(12)
        # 保留原始文件名，只在前面加doc_id防止冲突
        original_name = self._sanitize_filename(filename)
        safe_filename = f"{doc_id}_{original_name}"
        file_path = self.files_dir / session.user_id / safe_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # 创建文档记录
        doc = UploadedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=ext[1:],  # 去掉点
            file_path=str(file_path),
            file_size=len(file_content),
            status="pending"
        )
        
        session.documents.append(doc)
        session.updated_at = datetime.now().isoformat()
        self._save_session(session)
        
        return doc
    
    def add_document_from_url(self, session_id: str, url: str) -> Optional[UploadedDocument]:
        """从URL添加文档"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if len(session.documents) >= self.MAX_DOCUMENTS_PER_SESSION:
            return None
        
        # 验证URL
        if not self._is_valid_pdf_url(url):
            return None
        
        # 生成文档ID
        doc_id = secrets.token_hex(12)
        
        # 从URL提取文件名
        parsed = urlparse(url)
        filename = Path(parsed.path).name or f"document_{doc_id}.pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        safe_filename = f"{doc_id}_{self._sanitize_filename(filename)}"
        file_path = self.files_dir / session.user_id / safe_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文档记录（稍后下载）
        doc = UploadedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            file_path=str(file_path),
            original_url=url,
            status="pending"
        )
        
        session.documents.append(doc)
        session.updated_at = datetime.now().isoformat()
        self._save_session(session)
        
        return doc
    
    def download_url_document(self, doc: UploadedDocument) -> bool:
        """下载URL文档"""
        if not doc.original_url:
            return False
        
        try:
            import urllib.request
            import ssl
            
            # 创建SSL上下文（忽略证书验证，用于学术网站）
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                doc.original_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SciAgent/1.0)"}
            )
            
            with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                content = response.read()
                
            with open(doc.file_path, "wb") as f:
                f.write(content)
            
            doc.file_size = len(content)
            doc.status = "ready"
            return True
        except Exception as e:
            doc.status = "error"
            doc.error_message = str(e)
            return False
    
    def remove_document(self, session_id: str, doc_id: str) -> bool:
        """移除文档"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        for i, doc in enumerate(session.documents):
            if doc.doc_id == doc_id:
                # 删除文件
                if os.path.exists(doc.file_path):
                    os.remove(doc.file_path)
                session.documents.pop(i)
                session.updated_at = datetime.now().isoformat()
                self._save_session(session)
                return True
        
        return False
    
    def update_document_status(self, session_id: str, doc_id: str, 
                               status: str, **kwargs) -> bool:
        """更新文档状态"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        for doc in session.documents:
            if doc.doc_id == doc_id:
                doc.status = status
                for key, value in kwargs.items():
                    if hasattr(doc, key):
                        setattr(doc, key, value)
                session.updated_at = datetime.now().isoformat()
                self._save_session(session)
                return True
        
        return False
    
    def set_suggested_questions(self, session_id: str, questions: List[str]) -> bool:
        """设置推荐问题"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.suggested_questions = questions[:5]  # 最多5个
        session.updated_at = datetime.now().isoformat()
        self._save_session(session)
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有文档"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # 删除所有文件
        for doc in session.documents:
            if os.path.exists(doc.file_path):
                try:
                    os.remove(doc.file_path)
                except:
                    pass
        
        # 删除会话记录
        del self._sessions[session_id]
        file_path = self.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        return True
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        # 移除不安全字符
        safe = re.sub(r'[^\w\-_\.]', '_', filename)
        # 限制长度
        if len(safe) > 100:
            name, ext = os.path.splitext(safe)
            safe = name[:100-len(ext)] + ext
        return safe
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """验证是否是有效的PDF URL"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            # 常见的学术PDF来源
            valid_domains = [
                "arxiv.org", "openreview.net", "aclanthology.org",
                "proceedings.neurips.cc", "proceedings.mlr.press",
                "dl.acm.org", "ieee.org", "springer.com",
                "nature.com", "science.org", "pnas.org",
                "biorxiv.org", "medrxiv.org", "ssrn.com"
            ]
            # 检查是否是已知域名或以.pdf结尾
            if any(domain in parsed.netloc for domain in valid_domains):
                return True
            if parsed.path.lower().endswith(".pdf"):
                return True
            return False
        except:
            return False
