"""
认知记忆模块 - 文档上下文记忆模型

支持文档上下文管理:
- 文档元数据、编辑历史、内容摘要
- 文档编辑行为跟踪
- 文档上下文与用户会话关联
- 跨会话文档上下文自动关联与恢复
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DocumentFormat(Enum):
    """文档格式"""
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    TXT = "txt"
    PPT = "ppt"
    PPTX = "pptx"
    XLS = "xls"
    XLSX = "xlsx"
    MARKDOWN = "md"
    OTHER = "other"


class DocumentAction(Enum):
    """文档操作类型"""
    CREATE = "create"
    OPEN = "open"
    EDIT = "edit"
    SAVE = "save"
    CLOSE = "close"
    SHARE = "share"
    DELETE = "delete"


@dataclass
class DocumentMetadata:
    """文档元数据"""
    file_name: str = ""
    file_path: str = ""
    file_format: DocumentFormat = DocumentFormat.OTHER
    file_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    last_editor: str = ""
    page_count: int = 0
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_format": self.file_format.value,
            "file_size_bytes": self.file_size_bytes,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "author": self.author,
            "last_editor": self.last_editor,
            "page_count": self.page_count,
            "word_count": self.word_count,
        }


@dataclass
class DocumentEditRecord:
    """文档编辑记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    document_id: str = ""
    action: DocumentAction = DocumentAction.OPEN
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    duration_seconds: float = 0.0
    edit_scope: str = ""          # 编辑范围描述
    cursor_position: int = 0      # 编辑光标位置
    lines_changed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "document_id": self.document_id,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "edit_scope": self.edit_scope,
            "cursor_position": self.cursor_position,
            "lines_changed": self.lines_changed,
        }


@dataclass
class DocumentContext:
    """文档上下文 - 记录用户与文档的交互上下文"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    document: DocumentMetadata = field(default_factory=DocumentMetadata)

    # 内容摘要
    title: str = ""
    content_summary: str = ""          # 内容摘要（前200字符）
    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    # 编辑历史
    edit_history: list[DocumentEditRecord] = field(default_factory=list)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    total_edit_time: float = 0.0       # 累计编辑时间（秒）
    edit_count: int = 0

    # 会话关联
    associated_sessions: list[str] = field(default_factory=list)  # 关联的会话ID

    # 关联上下文
    related_documents: list[str] = field(default_factory=list)    # 关联文档ID
    related_projects: list[str] = field(default_factory=list)     # 关联项目
    related_teams: list[str] = field(default_factory=list)        # 关联团队

    # 元数据
    importance: int = 3                # 1-5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "document": self.document.to_dict(),
            "title": self.title,
            "content_summary": self.content_summary,
            "keywords": self.keywords,
            "topics": self.topics,
            "edit_history": [e.to_dict() for e in self.edit_history[-10:]],
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "total_edit_time": self.total_edit_time,
            "edit_count": self.edit_count,
            "associated_sessions": self.associated_sessions,
            "related_documents": self.related_documents,
            "related_projects": self.related_projects,
            "related_teams": self.related_teams,
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def add_edit_record(self, record: DocumentEditRecord):
        """添加编辑记录"""
        self.edit_history.append(record)
        self.edit_count += 1
        self.total_edit_time += record.duration_seconds
        self.last_accessed_at = record.timestamp
        if record.session_id and record.session_id not in self.associated_sessions:
            self.associated_sessions.append(record.session_id)

    def get_recent_edits(self, limit: int = 5) -> list[DocumentEditRecord]:
        """获取最近的编辑记录"""
        return sorted(
            self.edit_history,
            key=lambda x: x.timestamp,
            reverse=True,
        )[:limit]

    def is_related_to_session(self, session_id: str) -> bool:
        """检查是否与指定会话关联"""
        return session_id in self.associated_sessions


@dataclass
class DocumentContextQuery:
    """文档上下文查询请求"""
    user_id: str
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    max_results: int = 10
    time_range_days: Optional[int] = None
    sort_by: str = "last_accessed"  # "last_accessed", "edit_count", "importance"