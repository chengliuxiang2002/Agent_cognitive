"""
认知记忆模块 - 团队协作记忆模型

支持企业级团队协作场景:
- 团队共享记忆存储
- 个人记忆与团队记忆的隔离
- 权限管理（查看/编辑权限控制）
- 常用团队信息类型：会议室信息、项目文档路径、团队规范等
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TeamPermission(Enum):
    """团队权限级别"""
    VIEW = "view"      # 只读
    EDIT = "edit"      # 可编辑
    ADMIN = "admin"    # 可管理成员与权限


class TeamMemoryType(Enum):
    """团队记忆类型"""
    MEETING_ROOM = "meeting_room"          # 会议室信息
    PROJECT_DOC = "project_doc"            # 项目文档路径
    TEAM_NORM = "team_norm"                # 团队规范
    CONTACT_INFO = "contact_info"          # 联系人信息
    PROCESS_GUIDE = "process_guide"        # 流程指南
    GENERAL = "general"                    # 通用信息


@dataclass
class TeamMember:
    """团队成员"""
    user_id: str
    role: str = "member"                   # "leader", "member", "guest"
    permission: TeamPermission = TeamPermission.VIEW
    joined_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "permission": self.permission.value,
            "joined_at": self.joined_at.isoformat(),
        }


@dataclass
class TeamMemory:
    """团队记忆条目 - 团队级共享记忆的基本存储单元"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    team_id: str = ""
    title: str = ""
    memory_type: TeamMemoryType = TeamMemoryType.GENERAL
    content: dict[str, Any] = field(default_factory=dict)

    # 创建者与更新时间
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_by: str = ""
    updated_at: datetime = field(default_factory=datetime.now)

    # 关联标签
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # 重要性 (1-5)
    importance: int = 3

    # 访问控制
    is_public: bool = True           # 是否团队内公开
    allowed_members: list[str] = field(default_factory=list)  # 特定可见成员（is_public=False时）

    # 元数据
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "title": self.title,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "keywords": self.keywords,
            "importance": self.importance,
            "is_public": self.is_public,
            "allowed_members": self.allowed_members,
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass
class Team:
    """团队定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    department: str = ""
    members: list[TeamMember] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 团队元数据
    tags: list[str] = field(default_factory=list)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "department": self.department,
            "members": [m.to_dict() for m in self.members],
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    def has_member(self, user_id: str) -> bool:
        """检查用户是否为团队成员"""
        return any(m.user_id == user_id for m in self.members)

    def get_permission(self, user_id: str) -> Optional[TeamPermission]:
        """获取用户在团队中的权限"""
        for m in self.members:
            if m.user_id == user_id:
                return m.permission
        return None

    def can_edit(self, user_id: str) -> bool:
        """检查用户是否有编辑权限"""
        perm = self.get_permission(user_id)
        return perm in (TeamPermission.EDIT, TeamPermission.ADMIN)

    def can_admin(self, user_id: str) -> bool:
        """检查用户是否有管理员权限"""
        return self.get_permission(user_id) == TeamPermission.ADMIN


@dataclass
class TeamMemoryQuery:
    """团队记忆查询请求"""
    team_id: str
    user_id: str = ""                      # 查询者（用于权限校验）
    memory_types: list[TeamMemoryType] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_results: int = 20
    sort_by: str = "updated_at"            # "updated_at", "created_at", "importance"
    time_range_days: Optional[int] = None