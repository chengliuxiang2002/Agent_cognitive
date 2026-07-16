"""
认知记忆模块 - 短期记忆存储实现

基于内存字典的短期记忆存储，适用于当前会话上下文。
支持:
- 快速存取（O(1) 查询）
- 自动过期清理
- 容量限制
- 按重要性优先级淘汰

生产环境可替换为 Redis 实现，接口保持一致。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional

from .base import BaseMemoryStore
from ..models.memory import (
    MemoryItem,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryType,
)


class ShortTermMemoryStore(BaseMemoryStore):
    """短期记忆存储 - 基于 LRU 的内存存储"""

    def __init__(
        self,
        max_capacity: int = 500,
        ttl_seconds: int = 3600,  # 默认1小时过期
    ):
        self._store: OrderedDict[str, MemoryItem] = OrderedDict()
        self._max_capacity = max_capacity
        self._ttl_seconds = ttl_seconds

    async def store(self, item: MemoryItem) -> bool:
        """存储短期记忆，超过容量时按LRU淘汰"""
        # 淘汰过期条目
        await self._evict_expired()

        # 如果达到容量上限，淘汰最旧的
        if len(self._store) >= self._max_capacity:
            self._store.popitem(last=False)

        item.last_updated_at = datetime.now()
        self._store[item.id] = item
        self._store.move_to_end(item.id)  # 标记为最近使用
        return True

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """检索单条记忆并更新访问时间"""
        item = self._store.get(memory_id)
        if item:
            if await self._is_expired(item):
                del self._store[memory_id]
                return None
            item.last_accessed_at = datetime.now()
            item.access_count += 1
            self._store.move_to_end(memory_id)
        return item

    async def query(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """根据查询条件检索记忆"""
        start_time = time.time()
        await self._evict_expired()

        results: list[MemoryItem] = []

        for item in self._store.values():
            if not self._match_query(item, query):
                continue
            results.append(item)

        # 排序
        results = self._sort_results(results, query.sort_by)

        # 限制数量
        total = len(results)
        results = results[:query.max_results]

        retrieval_time = (time.time() - start_time) * 1000

        return MemoryRetrievalResult(
            query=query,
            items=results,
            total_found=total,
            retrieval_time_ms=retrieval_time,
        )

    async def update(self, item: MemoryItem) -> bool:
        """更新记忆条目"""
        if item.id in self._store:
            item.last_updated_at = datetime.now()
            self._store[item.id] = item
            self._store.move_to_end(item.id)
            return True
        return False

    async def delete(self, memory_id: str) -> bool:
        """删除记忆条目"""
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    async def delete_by_user(self, user_id: str) -> int:
        """删除指定用户的所有记忆"""
        to_delete = [
            mid for mid, item in self._store.items()
            if item.user_id == user_id
        ]
        for mid in to_delete:
            del self._store[mid]
        return len(to_delete)

    async def get_user_memories(
        self, user_id: str, memory_type: Optional[str] = None, limit: int = 100
    ) -> list[MemoryItem]:
        """获取指定用户的记忆"""
        results = []
        for item in self._store.values():
            if item.user_id != user_id:
                continue
            if memory_type and item.memory_type.value != memory_type:
                continue
            results.append(item)
            if len(results) >= limit:
                break
        return results

    async def cleanup_weak_memories(self, threshold: float = 0.1) -> int:
        """清理强度低于阈值的记忆"""
        to_delete = [
            mid for mid, item in self._store.items()
            if item.strength < threshold
        ]
        for mid in to_delete:
            del self._store[mid]
        return len(to_delete)

    async def _evict_expired(self) -> int:
        """淘汰过期条目"""
        to_delete = [
            mid for mid, item in self._store.items()
            if await self._is_expired(item)
        ]
        for mid in to_delete:
            del self._store[mid]
        return len(to_delete)

    async def _is_expired(self, item: MemoryItem) -> bool:
        """检查记忆是否过期"""
        elapsed = (datetime.now() - item.last_accessed_at).total_seconds()
        return elapsed > self._ttl_seconds

    def _match_query(self, item: MemoryItem, query: MemoryQuery) -> bool:
        """检查记忆条目是否匹配查询条件"""
        # 用户ID匹配
        if query.user_id and item.user_id != query.user_id:
            return False

        # 记忆类型匹配
        if query.memory_types and item.memory_type not in query.memory_types:
            return False

        # 强度阈值
        if item.strength < query.min_strength:
            return False

        # 重要性阈值
        if item.importance.value < query.min_importance.value:
            return False

        # 标签匹配（OR逻辑）
        if query.tags:
            if not any(tag in item.tags for tag in query.tags):
                return False

        # 关键词匹配
        if query.keywords:
            content_str = str(item.content).lower()
            if not any(kw.lower() in content_str for kw in query.keywords):
                return False

        # 时间范围
        if query.time_range_days is not None:
            cutoff = datetime.now() - timedelta(days=query.time_range_days)
            if item.created_at < cutoff:
                return False

        return True

    def _sort_results(
        self, items: list[MemoryItem], sort_by: str
    ) -> list[MemoryItem]:
        """对结果排序"""
        if sort_by == "recency":
            return sorted(items, key=lambda x: x.last_accessed_at, reverse=True)
        elif sort_by == "strength":
            return sorted(items, key=lambda x: x.strength, reverse=True)
        elif sort_by == "importance":
            return sorted(items, key=lambda x: x.importance.value, reverse=True)
        else:  # relevance: 综合强度+重要性+访问次数
            return sorted(
                items,
                key=lambda x: (
                    x.strength * 0.4
                    + (x.importance.value / 5) * 0.3
                    + min(x.access_count / 10, 1.0) * 0.3
                ),
                reverse=True,
            )

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def capacity(self) -> int:
        return self._max_capacity