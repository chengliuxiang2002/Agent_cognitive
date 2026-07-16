"""
认知记忆模块 - 记忆衰减与强化机制

模拟人类记忆特性:
- 艾宾浩斯遗忘曲线: 记忆强度随时间指数衰减
- 间隔重复效应: 每次访问/使用增强记忆强度
- 重要性加权: 高重要性记忆衰减更慢
- 情感关联: 与强烈情感关联的记忆更持久

数学公式:
  衰减: S(t) = S_0 * e^(-λ * t * (1 / importance_weight))
  强化: S_new = min(1.0, S_old + ΔS * (1 - S_old))
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from ..models.memory import MemoryItem, MemoryImportance


class MemoryDecayEngine:
    """记忆衰减引擎 - 管理记忆强度的衰减与强化"""

    # 重要性权重映射（重要性越高，衰减越慢）
    IMPORTANCE_WEIGHT = {
        MemoryImportance.CRITICAL: 5.0,
        MemoryImportance.HIGH: 3.0,
        MemoryImportance.MEDIUM: 1.5,
        MemoryImportance.LOW: 1.0,
        MemoryImportance.TRANSIENT: 0.5,
    }

    def __init__(
        self,
        base_decay_rate: float = 0.01,     # 基础衰减速率
        reinforcement_gain: float = 0.15,  # 每次访问强化增益
        decay_interval_hours: float = 1.0, # 衰减计算间隔
    ):
        self._base_decay_rate = base_decay_rate
        self._reinforcement_gain = reinforcement_gain
        self._decay_interval_hours = decay_interval_hours

    def calculate_decay(self, item: MemoryItem) -> float:
        """计算衰减后的记忆强度"""
        elapsed_hours = (
            datetime.now() - item.last_accessed_at
        ).total_seconds() / 3600.0

        if elapsed_hours <= 0:
            return item.strength

        # 重要性越高，有效衰减速率越低
        importance_weight = self.IMPORTANCE_WEIGHT.get(
            item.importance, 1.0
        )
        effective_decay_rate = self._base_decay_rate / importance_weight

        # 指数衰减: S(t) = S_0 * e^(-λ * t)
        decayed_strength = item.strength * math.exp(
            -effective_decay_rate * elapsed_hours
        )

        # 确保不低于最低强度
        return max(decayed_strength, item.min_strength)

    def apply_reinforcement(self, item: MemoryItem) -> float:
        """应用记忆强化（每次访问时调用）"""
        # S_new = S_old + ΔS * (1 - S_old)  → 越接近1.0，增长越慢
        delta = self._reinforcement_gain * (1.0 - item.strength)
        new_strength = min(1.0, item.strength + delta)
        return new_strength

    def apply_emotional_boost(
        self, item: MemoryItem, emotional_intensity: float
    ) -> float:
        """情感增强：高情感强度的记忆衰减更慢

        Args:
            emotional_intensity: 情感强度 0.0 ~ 1.0
        """
        boost = emotional_intensity * 0.2  # 最多提升 20%
        return min(1.0, item.strength + boost)

    def should_forget(self, item: MemoryItem) -> bool:
        """判断记忆是否应该被遗忘"""
        current_strength = self.calculate_decay(item)
        return current_strength < item.min_strength

    def update_memory_strength(
        self,
        item: MemoryItem,
        accessed: bool = False,
        emotional_intensity: float = 0.0,
    ) -> MemoryItem:
        """更新记忆条目的强度

        Args:
            item: 记忆条目
            accessed: 是否刚被访问（触发强化）
            emotional_intensity: 关联的情感强度

        Returns:
            更新后的记忆条目
        """
        # 先计算衰减
        decayed = self.calculate_decay(item)

        # 如果被访问，应用强化
        if accessed:
            item.strength = self.apply_reinforcement(item)
            item.access_count += 1
        else:
            item.strength = decayed

        # 应用情感增强
        if emotional_intensity > 0:
            item.strength = self.apply_emotional_boost(item, emotional_intensity)

        item.last_accessed_at = datetime.now()
        return item


class MemoryConsolidator:
    """记忆巩固器 - 将短期记忆转化为长期记忆"""

    def __init__(
        self,
        consolidation_threshold: float = 0.6,  # 强度阈值（超过则转化）
        min_access_count: int = 3,             # 最少访问次数
        min_strength_for_long_term: float = 0.5,
    ):
        self._threshold = consolidation_threshold
        self._min_access = min_access_count
        self._min_strength = min_strength_for_long_term

    def should_consolidate(self, item: MemoryItem) -> bool:
        """判断短期记忆是否应该转化为长期记忆"""
        return (
            item.strength >= self._threshold
            and item.access_count >= self._min_access
            and item.importance.value >= MemoryImportance.MEDIUM.value
        )

    def consolidate(
        self, item: MemoryItem, decay_engine: MemoryDecayEngine
    ) -> MemoryItem:
        """将短期记忆转化为长期记忆

        转化时:
        - 降低衰减速率（长期记忆衰减更慢）
        - 设置更高的最低强度阈值
        - 提升重要性等级（如果需要）
        """
        from ..models.memory import MemoryType

        item.memory_type = MemoryType.LONG_TERM
        item.decay_rate = item.decay_rate * 0.3  # 长期记忆衰减慢3倍
        item.min_strength = max(item.min_strength, 0.3)

        # 重新计算强度以反映新的衰减速率
        item.strength = decay_engine.calculate_decay(item)

        return item