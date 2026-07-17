"""
认知记忆模块 - 存储层抽象接口

定义记忆存储的统一接口，支持多种后端实现。
所有存储实现必须遵循此接口，确保模块的可替换性。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models.memory import MemoryItem, MemoryQuery, MemoryRetrievalResult, UserProfile, InteractionRecord, BehaviorPattern


class BaseMemoryStore(ABC):
    """记忆存储抽象基类"""

    @abstractmethod
    async def store(self, item: MemoryItem) -> bool:
        """存储一条记忆"""
        ...

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """根据ID检索单条记忆"""
        ...

    @abstractmethod
    async def query(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """根据查询条件检索记忆"""
        ...

    @abstractmethod
    async def update(self, item: MemoryItem) -> bool:
        """更新记忆条目"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除记忆条目"""
        ...

    @abstractmethod
    async def delete_by_user(self, user_id: str) -> int:
        """删除指定用户的所有记忆，返回删除数量"""
        ...

    @abstractmethod
    async def get_user_memories(
        self, user_id: str, memory_type: Optional[str] = None, limit: int = 100
    ) -> list[MemoryItem]:
        """获取指定用户的所有记忆"""
        ...

    @abstractmethod
    async def cleanup_weak_memories(self, threshold: float = 0.1) -> int:
        """清理强度低于阈值的记忆，返回清理数量"""
        ...


class BaseProfileStore(ABC):
    """用户画像存储抽象基类"""

    @abstractmethod
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        ...

    @abstractmethod
    async def save_profile(self, profile: UserProfile) -> bool:
        """保存或更新用户画像"""
        ...

    @abstractmethod
    async def delete_profile(self, user_id: str) -> bool:
        """删除用户画像"""
        ...


class BaseInteractionStore(ABC):
    """交互记录存储抽象基类"""

    @abstractmethod
    async def record(self, interaction: InteractionRecord) -> bool:
        """记录一次交互"""
        ...

    @abstractmethod
    async def get_recent(
        self, user_id: str, limit: int = 50
    ) -> list[InteractionRecord]:
        """获取最近的交互记录"""
        ...

    @abstractmethod
    async def get_by_session(
        self, session_id: str
    ) -> list[InteractionRecord]:
        """获取指定会话的所有交互"""
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """删除指定的交互记录"""
        ...


class BasePatternStore(ABC):
    """行为模式存储抽象基类"""

    @abstractmethod
    async def save_pattern(self, pattern: BehaviorPattern) -> bool:
        """保存行为模式"""
        ...

    @abstractmethod
    async def get_patterns(
        self, user_id: str, pattern_type: Optional[str] = None
    ) -> list[BehaviorPattern]:
        """获取用户的行为模式"""
        ...

    @abstractmethod
    async def get_pattern_by_name(
        self, user_id: str, pattern_name: str
    ) -> Optional[BehaviorPattern]:
        """根据名称获取特定模式"""
        ...

    @abstractmethod
    async def delete_pattern(self, pattern_id: str) -> bool:
        """删除指定的行为模式"""
        ...