"""
用户模型 - 用户认证和会话管理
"""
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class User:
    """用户模型"""
    user_id: str
    username: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(**data)


@dataclass
class Session:
    """会话模型"""
    session_id: str
    user_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = ""
    is_active: bool = True
    
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.now()


class UserManager:
    """用户管理器"""
    
    def __init__(self, data_dir: str = "data/users"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.sessions_file = self.data_dir / "sessions.json"
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._load_data()
    
    def _load_data(self):
        """加载用户和会话数据"""
        if self.users_file.exists():
            with open(self.users_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._users = {k: User.from_dict(v) for k, v in data.items()}
        
        if self.sessions_file.exists():
            with open(self.sessions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._sessions = {k: Session(**v) for k, v in data.items()}
    
    def _save_data(self):
        """保存数据"""
        with open(self.users_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._users.items()}, f, ensure_ascii=False, indent=2)
        
        with open(self.sessions_file, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self._sessions.items()}, f, ensure_ascii=False, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username: str, password: str) -> Optional[User]:
        """注册用户"""
        # 检查用户名是否已存在
        for user in self._users.values():
            if user.username == username:
                return None
        
        user_id = secrets.token_hex(16)
        user = User(
            user_id=user_id,
            username=username,
            password_hash=self._hash_password(password)
        )
        self._users[user_id] = user
        self._save_data()
        return user
    
    def login(self, username: str, password: str) -> Optional[Session]:
        """用户登录"""
        password_hash = self._hash_password(password)
        
        for user in self._users.values():
            if user.username == username and user.password_hash == password_hash:
                # 更新最后登录时间
                user.last_login = datetime.now().isoformat()
                
                # 创建会话（永久有效）
                session = Session(
                    session_id=secrets.token_hex(32),
                    user_id=user.user_id,
                    expires_at=""  # 空字符串表示永不过期
                )
                self._sessions[session.session_id] = session
                self._save_data()
                return session
        
        return None
    
    def logout(self, session_id: str) -> bool:
        """用户登出"""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False
            self._save_data()
            return True
        return False
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """验证会话"""
        session = self._sessions.get(session_id)
        if not session or not session.is_active or session.is_expired():
            return None
        return self._users.get(session.user_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
