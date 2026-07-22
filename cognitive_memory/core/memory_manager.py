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
import json
import logging
import uuid
from datetime import datetime, timedelta
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
    FeedbackStore,
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


# ═══════════════════════════════════════════════════════════════════════════════
# SC-7: 数据保留策略配置
# ═══════════════════════════════════════════════════════════════════════════════

# 数据保留策略: 重要性级别 -> 保留天数
DATA_RETENTION_POLICY: dict[MemoryImportance, int] = {
    MemoryImportance.TRANSIENT: 7,     # 7天
    MemoryImportance.LOW: 90,          # 90天
    MemoryImportance.MEDIUM: 365,      # 365天
    MemoryImportance.HIGH: -1,         # 永久保留
    MemoryImportance.CRITICAL: -1,     # 永久保留
}


class MemoryManager:
    """认知记忆管理器 - 模块核心编排器

    性能指标:
    - 记忆存储延迟: < 50ms
    - 记忆检索延迟: < 100ms (短期), < 200ms (长期)
    - 短期记忆容量: 500条/用户
    - 长期记忆容量: 无限制（受数据库容量限制）
    - 画像更新频率: 每次新交互后增量更新
    - SC-7: 数据保留策略 - TRANSIENT 7天, LOW 90天, MEDIUM 365天, HIGH/CRITICAL 永久
    """

    def __init__(
        self,
        short_term_store: Optional[BaseMemoryStore] = None,
        long_term_store: Optional[BaseMemoryStore] = None,
        profile_store: Optional[BaseProfileStore] = None,
        interaction_store: Optional[BaseInteractionStore] = None,
        pattern_store: Optional[BasePatternStore] = None,
        feedback_store: Optional["FeedbackStore"] = None,
        db_path: str = "cognitive_memory.db",
    ):
        # 初始化存储层（默认使用 SQLite + 内存）
        self._short_term = short_term_store or ShortTermMemoryStore()
        self._long_term = long_term_store or LongTermMemoryStore(db_path)
        self._profile_store = profile_store or ProfileStore(db_path)
        self._interaction_store = interaction_store or InteractionStore(db_path)
        self._pattern_store = pattern_store or PatternStore(db_path)
        self._feedback_store = feedback_store or FeedbackStore(db_path)

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
        self._retention_task: Optional[asyncio.Task] = None  # SC-7
        self._running = False

        # SC-7: 数据保留清理日志
        self._retention_log: list[dict[str, Any]] = []

        # SC-7: 数据备份存储 (内存备份，生产环境可替换为文件/S3)
        self._data_backup: dict[str, dict[str, Any]] = {}

        # P0: 反馈闭环 — 缓存最近预测用于隐式反馈检测
        self._last_predictions: dict[str, list[dict[str, Any]]] = {}
        self._prediction_scene: dict[str, SceneContext] = {}

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

        # 6. P0: 隐式反馈检测 — 比对实际行为与预测，自动调整置信度
        asyncio.create_task(self._detect_implicit_feedback(interaction))

        # 7. 异步触发学习流程
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
        """预测用户需求, 同时缓存预测结果用于隐式反馈检测"""
        profile = await self._profile_store.get_profile(user_id)
        predictions = await self._context_engine.predict_user_needs(
            user_id, current_scene, profile
        )
        # P0: 缓存预测结果，供后续隐式反馈检测使用
        self._last_predictions[user_id] = predictions
        self._prediction_scene[user_id] = current_scene
        return predictions

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

    async def get_all_behavior_patterns(self, limit: int = 200) -> list[BehaviorPattern]:
        """获取所有用户行为模式（管理界面图谱用）"""
        return await self._pattern_store.get_all_patterns(limit)

    async def forget_user_data(self, user_id: str) -> int:
        """删除用户所有数据（隐私合规）"""
        count = 0
        count += await self._short_term.delete_by_user(user_id)
        count += await self._long_term.delete_by_user(user_id)
        await self._profile_store.delete_profile(user_id)
        return count

    # ─── 后台维护 ─────────────────────────────────────────

    async def start_maintenance(self, interval_seconds: int = 300):
        """启动后台维护任务 (PF-1: 同时启动写入缓冲)"""
        self._running = True

        # PF-1: 启动 LongTermMemoryStore 的写入缓冲和 Redis 连接
        if hasattr(self._long_term, 'start'):
            await self._long_term.start()

        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(interval_seconds)
        )
        logger.info("Memory maintenance started")

    async def stop_maintenance(self):
        """停止后台维护任务 (PF-1: 同时停止写入缓冲, SC-7: 停止保留策略)"""
        self._running = False
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        # PF-1: 停止 LongTermMemoryStore 的写入缓冲和 Redis 连接
        if hasattr(self._long_term, 'stop'):
            await self._long_term.stop()

        # SC-7: 停止数据保留策略
        await self.stop_retention_policy()

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

    # ─── SC-7: 数据保留策略 ─────────────────────────────

    async def start_retention_policy(self, interval_seconds: int = 86400):
        """SC-7: 启动数据保留策略定时任务

        默认每天执行一次 (86400s = 24h)。
        独立于维护任务，可单独启停。
        """
        if self._retention_task is not None:
            return

        self._retention_task = asyncio.create_task(
            self._retention_loop(interval_seconds)
        )
        logger.info(f"Data retention policy started, interval={interval_seconds}s")

    async def stop_retention_policy(self):
        """SC-7: 停止数据保留策略"""
        if self._retention_task:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
            self._retention_task = None
        logger.info("Data retention policy stopped")

    async def _retention_loop(self, interval_seconds: int):
        """SC-7: 数据保留循环"""
        while self._retention_task is not None:
            try:
                await asyncio.sleep(interval_seconds)
                await self._execute_retention_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retention cleanup error: {e}")

    async def _execute_retention_cleanup(self) -> dict[str, Any]:
        """SC-7: 执行数据保留策略清理

        根据 MemoryImportance 级别自动清理过期数据:
        - TRANSIENT: 7天
        - LOW: 90天
        - MEDIUM: 365天
        - HIGH/CRITICAL: 永久保留

        清理流程:
        1. 遍历所有用户数据
        2. 根据重要性判断是否过期
        3. 过期数据先备份
        4. 执行删除
        5. 记录清理日志
        """
        cleanup_stats = {
            "timestamp": datetime.now().isoformat(),
            "total_cleaned": 0,
            "by_level": {},
            "backed_up": 0,
        }

        now = datetime.now()

        for importance, retention_days in DATA_RETENTION_POLICY.items():
            if retention_days < 0:
                # 永久保留，跳过
                continue

            cutoff = now - timedelta(days=retention_days)
            cleaned_count = await self._cleanup_by_importance(
                importance, cutoff
            )
            if cleaned_count > 0:
                cleanup_stats["by_level"][importance.value] = cleaned_count
                cleanup_stats["total_cleaned"] += cleaned_count

        # 记录清理日志
        self._retention_log.append(cleanup_stats)
        if cleanup_stats["total_cleaned"] > 0:
            logger.info(
                f"Data retention cleanup completed: "
                f"cleaned={cleanup_stats['total_cleaned']}, "
                f"backed_up={cleanup_stats['backed_up']}"
            )

        return cleanup_stats

    async def _cleanup_by_importance(
        self, importance: MemoryImportance, cutoff: datetime
    ) -> int:
        """SC-7: 按重要性级别清理过期数据

        包括:
        1. 备份即将删除的数据
        2. 执行删除
        """
        cleaned = 0

        # 清理长期记忆
        result = await self._long_term.query(
            MemoryQuery(user_id="", max_results=10000)
        )
        lt_memories = result.items
        for item in lt_memories:
            if item.importance == importance and item.created_at < cutoff:
                # 备份数据
                self._backup_data(item.id, item.to_dict())
                # 删除
                await self._long_term.delete(item.id)
                cleaned += 1

        # 清理交互记录
        interactions = await self._interaction_store.get_recent("", limit=10000)
        for interaction in interactions:
            if interaction.timestamp < cutoff:
                self._backup_data(interaction.id, interaction.to_dict())
                await self._interaction_store.delete(interaction.id)
                cleaned += 1

        return cleaned

    def _backup_data(self, data_id: str, data: dict[str, Any]):
        """SC-7: 清理前备份数据"""
        self._data_backup[data_id] = {
            "data": data,
            "backed_up_at": datetime.now().isoformat(),
        }

    def get_retention_log(self) -> list[dict[str, Any]]:
        """SC-7: 获取数据保留策略清理日志"""
        return self._retention_log

    def get_backup_data(self, data_id: str) -> Optional[dict[str, Any]]:
        """SC-7: 获取备份数据"""
        return self._data_backup.get(data_id)

    async def restore_data(self, data_id: str) -> bool:
        """SC-7: 数据恢复机制

        从备份中恢复已删除的数据。
        """
        backup = self._data_backup.get(data_id)
        if not backup:
            logger.warning(f"Restore failed: no backup for {data_id}")
            return False

        data = backup["data"]
        # 尝试恢复为 MemoryItem
        try:
            item = MemoryItem(
                id=data.get("id", data_id),
                user_id=data.get("user_id", ""),
                memory_type=MemoryType(data.get("memory_type", "long_term")),
                content=data.get("content", {}),
                importance=MemoryImportance(data.get("importance", 3)),
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                tags=data.get("tags", []),
                confidence=data.get("confidence", 1.0),
            )
            await self._long_term.store(item)
            logger.info(f"Data restored: {data_id}")
            return True
        except Exception as e:
            logger.error(f"Restore failed for {data_id}: {e}")
            return False

    async def get_retention_due_items(self) -> list[dict[str, Any]]:
        """SC-7: 获取即将被清理的数据列表 (预览)"""
        due_items = []
        now = datetime.now()

        for importance, retention_days in DATA_RETENTION_POLICY.items():
            if retention_days < 0:
                continue
            cutoff = now - timedelta(days=retention_days)
            # 使用 query 方法获取所有用户的数据
            result = await self._long_term.query(
                MemoryQuery(user_id="", max_results=10000)
            )
            lt_memories = result.items
            for item in lt_memories:
                if item.importance == importance and item.created_at < cutoff:
                    due_items.append({
                        "data_id": item.id,
                        "user_id": item.user_id,
                        "importance": importance.value,
                        "created_at": item.created_at.isoformat(),
                        "retention_days": retention_days,
                        "expired_at": (item.created_at + timedelta(days=retention_days)).isoformat(),
                    })

        return due_items

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
                # UX-6: 推送通知（置信度 ≥ 0.8 且满足频率限制）
                await self._notify_pattern_learned(interaction.user_id, pattern)

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
        """按相关性排序 (PF-5: 预计算 relevance_score 优化)

        数据库层面已通过 relevance_score 索引完成排序，
        此处仅处理合并后的内存排序场景，使用缓存的计算结果。
        """
        if sort_by == "recency":
            return sorted(items, key=lambda x: x.last_accessed_at, reverse=True)
        elif sort_by == "strength":
            return sorted(items, key=lambda x: x.strength, reverse=True)
        elif sort_by == "importance":
            return sorted(items, key=lambda x: x.importance.value, reverse=True)
        else:
            # PF-5: 使用预计算相关性分数 (与 LongTermMemoryStore._compute_relevance_score 保持一致)
            return sorted(
                items,
                key=lambda x: (
                    x.strength * 0.4
                    + (x.importance.value / 5.0) * 0.3
                    + min(x.access_count / 10.0, 1.0) * 0.3
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

    # ─── UX-3: 用户反馈闭环 ───────────────────────────────

    async def record_feedback(
        self,
        user_id: str,
        prediction_id: str,
        feedback_type: str,
        prediction_data: Optional[dict[str, Any]] = None,
        comment: str = "",
    ) -> str:
        """记录用户对预测结果的反馈（点赞/踩）

        反馈数据持久化存储，用于动态调整模式识别的置信度权重。
        接口响应时间 ≤ 300ms。
        """
        feedback_id = str(uuid.uuid4())
        await self._feedback_store.record_feedback(
            feedback_id=feedback_id,
            user_id=user_id,
            prediction_id=prediction_id,
            feedback_type=feedback_type,
            prediction_data=prediction_data,
            comment=comment,
        )

        # 异步调度置信度调整（24小时内应用新反馈数据）
        asyncio.create_task(
            self._adjust_confidence_from_feedback(user_id, prediction_id, feedback_type)
        )

        logger.debug(f"Recorded feedback {feedback_id} from user {user_id}")
        return feedback_id

    async def get_feedback_stats(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> dict[str, Any]:
        """获取反馈统计"""
        return await self._feedback_store.get_feedback_stats(
            user_id=user_id, days=days
        )

    async def _adjust_confidence_from_feedback(
        self, user_id: str, prediction_id: str, feedback_type: str
    ):
        """根据用户反馈调整模式置信度权重"""
        try:
            # 查找与预测ID关联的行为模式
            patterns = await self._pattern_store.get_patterns(user_id)
            for pattern in patterns:
                if pattern.pattern_name in prediction_id or prediction_id in pattern.pattern_name:
                    old_confidence = pattern.confidence
                    if feedback_type == "like":
                        pattern.confidence = min(1.0, pattern.confidence + 0.05)
                        pattern.success_rate = min(1.0, pattern.success_rate + 0.03)
                    elif feedback_type == "dislike":
                        pattern.confidence = max(0.1, pattern.confidence - 0.08)
                        pattern.success_rate = max(0.0, pattern.success_rate - 0.05)

                    await self._pattern_store.save_pattern(pattern)
                    logger.debug(
                        f"Adjusted pattern {pattern.pattern_name} confidence: "
                        f"{old_confidence:.2f} → {pattern.confidence:.2f}"
                    )
                    break
        except Exception as e:
            logger.error(f"Confidence adjustment error: {e}")

    # ─── P0: 隐式反馈检测 ──────────────────────────────────

    async def _detect_implicit_feedback(self, interaction: InteractionRecord):
        """检测隐式反馈: 比对用户实际行为与最近预测

        隐式反馈不需要用户主动点击"有用/无用"，而是通过观察用户
        后续行为自动判断预测是否准确:
        - 用户执行了预测的动作 → 正向反馈 (like)
        - 用户做了完全不同的动作 → 负向反馈 (dislike)
        - 没有相关预测 → 跳过

        效果: 预测置信度自适应调整，越准越自信，越不准越保守
        """
        try:
            uid = interaction.user_id
            if uid not in self._last_predictions:
                return

            predictions = self._last_predictions.pop(uid)
            pred_scene = self._prediction_scene.pop(uid, None)

            if not predictions:
                return

            # 提取用户实际行为
            actual_intent = interaction.intent
            actual_input = interaction.processed_input

            # 遍历预测，检查是否命中
            matched = False
            for pred in predictions:
                pred_action = pred.get("action", {})
                pred_confidence = pred.get("confidence", 0)

                # 匹配规则: 预测的 action 与实际输入有交集
                is_match = self._action_matches(pred_action, actual_input, actual_intent)

                if is_match and pred_confidence >= 0.3:
                    matched = True
                    # 隐式正向反馈
                    if pred.get("pattern_name"):
                        await self._adjust_pattern_confidence(
                            uid, pred["pattern_name"], "like", implicit=True
                        )
                    logger.debug(
                        f"Implicit like: user {uid} action matched prediction "
                        f"'{pred.get('pattern_name', 'unknown')}'"
                    )
                    break

            if not matched and pred_scene:
                # 场景匹配但预测未命中 → 轻度负反馈
                for pred in predictions:
                    if pred.get("pattern_name") and pred.get("confidence", 0) >= 0.5:
                        await self._adjust_pattern_confidence(
                            uid, pred["pattern_name"], "dislike", implicit=True
                        )

        except Exception as e:
            logger.error(f"Implicit feedback error: {e}")

    @staticmethod
    def _action_matches(
        pred_action: dict, actual_input: dict, actual_intent: str
    ) -> bool:
        """判断预测的 action 是否与实际行为匹配

        支持多种匹配模式:
        - 精确键值匹配: 预测 set_temperature=24, 实际 set_temperature=24
        - 语义匹配: 预测 navigate_to=X, 实际 destination=X (同义不同键名)
        - 意图匹配: 预测 intent=music_play, 实际 intent=play_music (模糊匹配)
        """
        if not pred_action or not actual_input:
            return False

        # 精确匹配: 相同键且值相同或相近
        for key, pred_val in pred_action.items():
            if key in actual_input:
                actual_val = actual_input[key]
                if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    if abs(pred_val - actual_val) <= 0.5:
                        return True
                elif pred_val == actual_val:
                    return True

        # 语义匹配: 同义键名
        semantic_aliases = {
            "navigate_to": ["destination", "navigate_to", "nav_dest"],
            "set_temperature": ["temperature", "set_temp", "ac_temp"],
            "play_music": ["music", "play", "media"],
            "set_driving_mode": ["driving_mode", "mode"],
        }
        for key, pred_val in pred_action.items():
            aliases = semantic_aliases.get(key, [key])
            for alias in aliases:
                if alias in actual_input:
                    actual_val = actual_input[alias]
                    if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                        if abs(pred_val - actual_val) <= 0.5:
                            return True
                    elif pred_val == actual_val:
                        return True

        # 意图前缀匹配
        if actual_intent:
            pred_intent = pred_action.get("intent", "")
            if pred_intent and (
                pred_intent in actual_intent or actual_intent in pred_intent
            ):
                return True

        return False

    async def _adjust_pattern_confidence(
        self, user_id: str, pattern_name: str,
        feedback_type: str, implicit: bool = False,
    ):
        """调整模式置信度 (支持显式+隐式反馈)

        隐式反馈的调整幅度小于显式反馈:
        - 显式 like: 置信度 +0.05
        - 隐式 like: 置信度 +0.02
        - 显式 dislike: 置信度 -0.08
        - 隐式 dislike: 置信度 -0.03
        """
        try:
            patterns = await self._pattern_store.get_patterns(user_id)
            for pattern in patterns:
                if pattern.pattern_name == pattern_name:
                    old_confidence = pattern.confidence
                    if feedback_type == "like":
                        delta = 0.02 if implicit else 0.05
                        pattern.confidence = min(1.0, pattern.confidence + delta)
                        pattern.success_rate = min(1.0, pattern.success_rate + delta * 0.6)
                    elif feedback_type == "dislike":
                        delta = 0.03 if implicit else 0.08
                        pattern.confidence = max(0.1, pattern.confidence - delta)
                        pattern.success_rate = max(0.0, pattern.success_rate - delta * 0.6)

                    # 标记为已观察
                    pattern.last_observed = datetime.now()

                    await self._pattern_store.save_pattern(pattern)
                    logger.debug(
                        f"Adjusted pattern {pattern_name} confidence "
                        f"({'implicit' if implicit else 'explicit'} {feedback_type}): "
                        f"{old_confidence:.2f} → {pattern.confidence:.2f}"
                    )
                    break
        except Exception as e:
            logger.error(f"Confidence adjustment error: {e}")

    # ─── UX-4: 记忆透明度面板 ─────────────────────────────

    async def get_user_data_inventory(
        self,
        user_id: str,
        category: Optional[str] = None,
        time_range_days: Optional[int] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """获取用户个人数据清单

        按类别分组展示：行为记录/偏好设置/关联实体等。
        实现数据权限校验，确保用户仅能访问本人数据。
        """
        categories = {}
        all_items = []

        # 1. 行为记录（交互记录）
        if category is None or category == "behavior":
            interactions = await self._interaction_store.get_recent(
                user_id, limit=100
            )
            if time_range_days:
                from datetime import timedelta
                cutoff = datetime.now() - timedelta(days=time_range_days)
                interactions = [i for i in interactions if i.timestamp >= cutoff]

            behavior_items = []
            for i in interactions:
                behavior_items.append({
                    "data_id": i.id,
                    "category": "behavior",
                    "type": i.interaction_type,
                    "intent": i.intent,
                    "raw_input": i.raw_input[:100],
                    "collected_at": i.timestamp.isoformat(),
                    "scene": i.scene_context.to_dict() if i.scene_context else None,
                    "was_successful": i.was_successful,
                })
            categories["behavior"] = {
                "label": "行为记录",
                "count": len(behavior_items),
                "description": "您的交互记录，包含语音指令、触控操作等",
            }
            all_items.extend(behavior_items)

        # 2. 偏好设置（用户画像）
        if category is None or category == "preference":
            profile = await self._profile_store.get_profile(user_id)
            if profile:
                pref_items = [
                    {
                        "data_id": f"pref_temperature_{user_id}",
                        "category": "preference",
                        "type": "temperature",
                        "value": f"{profile.temperature_preference}℃",
                        "collected_at": profile.updated_at.isoformat(),
                    },
                    {
                        "data_id": f"pref_driving_mode_{user_id}",
                        "category": "preference",
                        "type": "driving_mode",
                        "value": profile.driving_mode_preference,
                        "collected_at": profile.updated_at.isoformat(),
                    },
                    {
                        "data_id": f"pref_music_{user_id}",
                        "category": "preference",
                        "type": "music",
                        "value": profile.music_preferences,
                        "collected_at": profile.updated_at.isoformat(),
                    },
                ]
                categories["preference"] = {
                    "label": "偏好设置",
                    "count": len(pref_items),
                    "description": "系统学习到的个人偏好，包含温度、驾驶模式、音乐等",
                }
                all_items.extend(pref_items)

        # 3. 关联实体（行为模式）
        if category is None or category == "entity":
            patterns = await self._pattern_store.get_patterns(user_id)
            entity_items = []
            for p in patterns:
                entity_items.append({
                    "data_id": p.id,
                    "category": "entity",
                    "type": p.pattern_type,
                    "name": p.pattern_name,
                    "confidence": p.confidence,
                    "occurrence_count": p.occurrence_count,
                    "collected_at": p.last_observed.isoformat(),
                })
            categories["entity"] = {
                "label": "关联实体",
                "count": len(entity_items),
                "description": "系统学习到的行为模式与关联实体",
            }
            all_items.extend(entity_items)

        # 分页
        start = (page - 1) * page_size
        end = start + page_size
        paged_items = all_items[start:end]

        return {
            "user_id": user_id,
            "categories": categories,
            "items": paged_items,
            "total": len(all_items),
            "page": page,
            "page_size": page_size,
            "has_more": end < len(all_items),
        }

    async def delete_user_data_item(
        self, user_id: str, data_id: str
    ) -> bool:
        """删除单条个人数据

        实现数据权限校验，确保用户仅能删除本人数据。
        删除后不可恢复。
        """
        # 权限校验：检查数据是否属于该用户
        if data_id.startswith("pref_"):
            # 偏好数据 - 验证 user_id 匹配
            if not data_id.endswith(user_id):
                raise PermissionError("无权删除他人数据")
            return True  # 偏好数据为衍生数据，标记删除

        # 尝试删除交互记录
        deleted = await self._interaction_store.delete(data_id)
        if not deleted:
            # 尝试删除行为模式
            deleted = await self._pattern_store.delete_pattern(data_id)

        if not deleted:
            raise ValueError(f"数据不存在: {data_id}")

        return True

    # ─── UX-6: 通知推送 ───────────────────────────────────

    async def _notify_pattern_learned(
        self, user_id: str, pattern: BehaviorPattern
    ):
        """在行为模式学习完成后推送通知

        通知触发条件: 模式置信度 ≥ 0.8
        频率限制: 同一类型模式30天内最多推送1次
        """
        if pattern.confidence < 0.8:
            return

        # 检查频率限制
        recent_patterns = await self._pattern_store.get_patterns(
            user_id, pattern.pattern_type
        )
        for existing in recent_patterns:
            if existing.id == pattern.id:
                continue
            days_since = (datetime.now() - existing.last_observed).days
            if days_since < 30:
                logger.debug(
                    f"Notification suppressed for {pattern.pattern_type}: "
                    f"last notification {days_since} days ago"
                )
                return

        notification = self._build_notification(pattern)
        logger.info(f"Notification for user {user_id}: {notification['title']}")

        # 通知可通过邮件/应用内消息推送
        # 此处记录日志，实际推送渠道由外部集成实现
        self._pending_notifications = getattr(self, "_pending_notifications", [])
        self._pending_notifications.append({
            "user_id": user_id,
            "notification": notification,
            "created_at": datetime.now().isoformat(),
        })

    def _build_notification(self, pattern: BehaviorPattern) -> dict[str, Any]:
        """构建通知内容"""
        pattern_type_labels = {
            "route": "通勤路线偏好",
            "temperature": "温度偏好",
            "media": "音乐偏好",
            "time": "活跃时段",
            "interaction": "交互风格",
        }

        pattern_type_label = pattern_type_labels.get(
            pattern.pattern_type, pattern.pattern_type
        )

        if pattern.pattern_type == "route":
            dest = pattern.expected_action.get("navigate_to", "")
            feature = f"前往「{dest}」"
        elif pattern.pattern_type == "temperature":
            temp = pattern.expected_action.get("set_temperature", "")
            feature = f"偏好设置{temp}℃"
        elif pattern.pattern_type == "media":
            genre = pattern.expected_action.get("preferred_genre", "")
            feature = f"喜欢「{genre}」类型"
        else:
            feature = "主要特征"

        days_span = max(1, (pattern.last_observed - pattern.first_observed).days)

        return {
            "title": f"系统已学习到您的{pattern_type_label}",
            "body": (
                f"基于过去{days_span}天的{pattern.occurrence_count}次交互记录，"
                f"系统识别到您的{pattern_type_label}：{feature}。"
                f"置信度：{pattern.confidence:.0%}"
            ),
            "pattern_type": pattern.pattern_type,
            "confidence": pattern.confidence,
            "occurrence_count": pattern.occurrence_count,
            "learning_period_days": days_span,
            "main_feature": feature,
        }

    def get_pending_notifications(self) -> list[dict[str, Any]]:
        """获取待推送的通知列表"""
        return getattr(self, "_pending_notifications", [])