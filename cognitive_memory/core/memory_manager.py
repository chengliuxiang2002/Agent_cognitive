"""
认知记忆模块 - 记忆管理器（核心编排器）

MemoryManager 是认知记忆模块的中央编排器，负责:
- 协调短期/长期记忆存储
- 管理记忆生命周期（编码→存储→衰减→巩固→遗忘）
- 调度用户画像构建和行为模式学习
- 提供统一的外部接口

模块间交互流程:
  InteractionRecord → MemoryEncoder → MemoryItem
       ↓                                     ↓
  UserProfileBuilder ← BehaviorPatterns ← PatternLearner
       ↓                                     ↓
  ProfileStore                         MemoryStore (Short/Long)
       ↓                                     ↓
  ContextEngine ←───────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from ..models.memory import (
    MemoryItem,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryType,
    MemoryImportance,
    UserProfile,
    InteractionRecord,
    SceneContext,
    BehaviorPattern,
)
from ..storage import (
    BaseMemoryStore,
    BaseProfileStore,
    BaseInteractionStore,
    BasePatternStore,
    ShortTermMemoryStore,
    LongTermMemoryStore,
    ProfileStore,
    InteractionStore,
    PatternStore,
)
from ..learner import (
    MemoryDecayEngine,
    MemoryConsolidator,
    PatternLearner,
    MemoryEncoder,
)
from .user_profile_builder import UserProfileBuilder
from .context_engine import ContextEngine

logger = logging.getLogger(__name__)


class MemoryManager:
    """认知记忆管理器 - 模块核心编排器

    性能指标:
    - 记忆存储延迟: < 50ms
    - 记忆检索延迟: < 100ms (短期), < 200ms (长期)
    - 短期记忆容量: 500条/用户
    - 长期记忆容量: 无限制（受数据库容量限制）
    - 画像更新频率: 每次新交互后增量更新
    """

    def __init__(
        self,
        short_term_store: Optional[BaseMemoryStore] = None,
        long_term_store: Optional[BaseMemoryStore] = None,
        profile_store: Optional[BaseProfileStore] = None,
        interaction_store: Optional[BaseInteractionStore] = None,
        pattern_store: Optional[BasePatternStore] = None,
        db_path: str = "cognitive_memory.db",
    ):
        # 初始化存储层（默认使用 SQLite + 内存）
        self._short_term = short_term_store or ShortTermMemoryStore()
        self._long_term = long_term_store or LongTermMemoryStore(db_path)
        self._profile_store = profile_store or ProfileStore(db_path)
        self._interaction_store = interaction_store or InteractionStore(db_path)
        self._pattern_store = pattern_store or PatternStore(db_path)

        # 初始化学习引擎
        self._decay_engine = MemoryDecayEngine()
        self._consolidator = MemoryConsolidator()
        self._pattern_learner = PatternLearner()
        self._memory_encoder = MemoryEncoder()

        # 初始化核心组件
        self._profile_builder = UserProfileBuilder(
            self._profile_store, self._interaction_store
        )
        self._context_engine = ContextEngine(
            self._long_term, self._pattern_store
        )

        # 后台任务
        self._maintenance_task: Optional[asyncio.Task] = None
        self._running = False

    # ─── 公共接口 ─────────────────────────────────────────

    async def record_interaction(
        self, interaction: InteractionRecord
    ) -> str:
        """记录一次用户交互，触发完整的记忆处理流程

        Args:
            interaction: 用户交互记录

        Returns:
            生成的记忆条目ID
        """
        # 1. 保存交互记录
        await self._interaction_store.record(interaction)
        logger.debug(f"Recorded interaction: {interaction.id}")

        # 2. 编码为记忆条目
        memory_item = self._memory_encoder.encode_interaction(
            interaction, interaction.user_id
        )

        # 3. 存储到短期记忆
        await self._short_term.store(memory_item)

        # 4. 同时存储到长期记忆（情景记忆）
        await self._long_term.store(memory_item)

        # 5. 如果有关联场景，也存储场景记忆
        if interaction.scene_context:
            scene_memory = self._memory_encoder.encode_scene(
                interaction.scene_context, interaction.user_id
            )
            await self._short_term.store(scene_memory)
            await self._long_term.store(scene_memory)

        # 6. 异步触发学习流程
        asyncio.create_task(self._learn_from_interaction(interaction))

        return memory_item.id

    async def store_memory(self, item: MemoryItem) -> str:
        """直接存储一条记忆"""
        if item.memory_type == MemoryType.SHORT_TERM:
            await self._short_term.store(item)
        else:
            await self._long_term.store(item)
        return item.id

    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """检索单条记忆（先查短期，再查长期）"""
        item = await self._short_term.retrieve(memory_id)
        if item:
            return item
        return await self._long_term.retrieve(memory_id)

    async def query_memories(
        self,
        user_id: str,
        memory_types: Optional[list[MemoryType]] = None,
        context: Optional[SceneContext] = None,
        tags: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        max_results: int = 20,
        sort_by: str = "relevance",
        time_range_days: Optional[int] = None,
    ) -> MemoryRetrievalResult:
        """综合查询记忆（短期 + 长期）"""
        query = MemoryQuery(
            user_id=user_id,
            memory_types=memory_types or [],
            context=context,
            tags=tags or [],
            keywords=keywords or [],
            max_results=max_results,
            sort_by=sort_by,
            time_range_days=time_range_days,
        )

        # 先查短期记忆
        short_result = await self._short_term.query(query)

        # 再查长期记忆
        long_result = await self._long_term.query(query)

        # 合并去重
        seen_ids = set()
        merged_items = []
        for item in short_result.items + long_result.items:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                merged_items.append(item)

        # 排序并限制数量
        merged_items = self._sort_by_relevance(merged_items, sort_by)
        merged_items = merged_items[:max_results]

        return MemoryRetrievalResult(
            query=query,
            items=merged_items,
            total_found=len(merged_items),
            retrieval_time_ms=short_result.retrieval_time_ms + long_result.retrieval_time_ms,
        )

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return await self._profile_store.get_profile(user_id)

    async def build_user_profile(self, user_id: str) -> UserProfile:
        """构建或更新用户画像"""
        return await self._profile_builder.build_profile(user_id)

    async def get_context_aware_memories(
        self,
        user_id: str,
        current_scene: SceneContext,
        max_results: int = 10,
    ) -> MemoryRetrievalResult:
        """获取与当前场景相关的记忆"""
        return await self._context_engine.get_context_aware_memories(
            user_id, current_scene, max_results
        )

    async def predict_user_needs(
        self,
        user_id: str,
        current_scene: SceneContext,
    ) -> list[dict[str, Any]]:
        """预测用户需求"""
        profile = await self._profile_store.get_profile(user_id)
        return await self._context_engine.predict_user_needs(
            user_id, current_scene, profile
        )

    async def detect_scene_change(
        self, previous: SceneContext, current: SceneContext
    ) -> dict[str, Any]:
        """检测场景变化"""
        return self._context_engine.detect_scene_change(previous, current)

    async def get_behavior_patterns(
        self, user_id: str, pattern_type: Optional[str] = None
    ) -> list[BehaviorPattern]:
        """获取用户行为模式"""
        return await self._pattern_store.get_patterns(user_id, pattern_type)

    async def forget_user_data(self, user_id: str) -> int:
        """删除用户所有数据（隐私合规）"""
        count = 0
        count += await self._short_term.delete_by_user(user_id)
        count += await self._long_term.delete_by_user(user_id)
        await self._profile_store.delete_profile(user_id)
        return count

    # ─── 后台维护 ─────────────────────────────────────────

    async def start_maintenance(self, interval_seconds: int = 300):
        """启动后台维护任务"""
        self._running = True
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(interval_seconds)
        )
        logger.info("Memory maintenance started")

    async def stop_maintenance(self):
        """停止后台维护任务"""
        self._running = False
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory maintenance stopped")

    async def _maintenance_loop(self, interval_seconds: int):
        """后台维护循环"""
        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                await self._run_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    async def _run_maintenance(self):
        """执行维护任务"""
        # 1. 衰减检查：清理过于虚弱的短期记忆
        st_cleaned = await self._short_term.cleanup_weak_memories(threshold=0.05)
        lt_cleaned = await self._long_term.cleanup_weak_memories(threshold=0.05)

        if st_cleaned or lt_cleaned:
            logger.info(
                f"Memory cleanup: {st_cleaned} short-term, {lt_cleaned} long-term"
            )

        # 2. 短期记忆巩固检查
        all_st_memories = await self._short_term.get_user_memories("", limit=1000)
        for item in all_st_memories:
            # 应用衰减
            item = self._decay_engine.update_memory_strength(item)
            await self._short_term.update(item)

            # 检查是否需要巩固为长期记忆
            if self._consolidator.should_consolidate(item):
                consolidated = self._consolidator.consolidate(item, self._decay_engine)
                await self._long_term.store(consolidated)
                logger.debug(f"Consolidated memory: {item.id}")

    # ─── 内部方法 ─────────────────────────────────────────

    async def _learn_from_interaction(self, interaction: InteractionRecord):
        """从交互中异步学习"""
        try:
            # 获取最近的交互记录
            recent = await self._interaction_store.get_recent(
                interaction.user_id, limit=50
            )

            # 获取或创建用户画像
            profile = await self._profile_store.get_profile(interaction.user_id)
            if profile is None:
                profile = UserProfile(user_id=interaction.user_id)

            # 学习行为模式
            patterns = self._pattern_learner.learn_all(recent, profile)

            # 保存新模式
            for pattern in patterns:
                await self._pattern_store.save_pattern(pattern)

            # 更新用户画像
            if patterns:
                await self._profile_builder.update_from_patterns(
                    interaction.user_id, patterns
                )

            logger.debug(
                f"Learned {len(patterns)} patterns for user {interaction.user_id}"
            )

        except Exception as e:
            logger.error(f"Learning error: {e}")

    def _sort_by_relevance(
        self, items: list[MemoryItem], sort_by: str
    ) -> list[MemoryItem]:
        """按相关性排序"""
        if sort_by == "recency":
            return sorted(items, key=lambda x: x.last_accessed_at, reverse=True)
        elif sort_by == "strength":
            return sorted(items, key=lambda x: x.strength, reverse=True)
        elif sort_by == "importance":
            return sorted(items, key=lambda x: x.importance.value, reverse=True)
        else:
            return sorted(
                items,
                key=lambda x: (
                    x.strength * 0.4
                    + (x.importance.value / 5) * 0.3
                    + min(x.access_count / 10, 1.0) * 0.3
                ),
                reverse=True,
            )

    # ─── 统计信息 ─────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """获取记忆系统统计信息"""
        return {
            "short_term_memories": self._short_term.size,
            "short_term_capacity": self._short_term.capacity,
            "timestamp": datetime.now().isoformat(),
        }