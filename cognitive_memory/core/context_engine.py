"""
认知记忆模块 - 上下文感知引擎

根据当前场景和用户历史记忆，提供上下文感知的记忆检索。
核心功能:
- 场景相似度匹配
- 基于上下文的记忆推荐
- 主动预测用户需求
- 场景切换检测
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from ..models.memory import (
    MemoryItem,
    MemoryQuery,
    MemoryRetrievalResult,
    SceneContext,
    UserProfile,
    BehaviorPattern,
    MemoryType,
    MemoryImportance,
)
from ..storage.base import BaseMemoryStore, BasePatternStore


class ContextEngine:
    """上下文感知引擎"""

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        pattern_store: BasePatternStore,
    ):
        self._memory_store = memory_store
        self._pattern_store = pattern_store

    def calculate_context_similarity(
        self, scene_a: SceneContext, scene_b: SceneContext
    ) -> float:
        """计算两个场景之间的相似度

        使用加权多维相似度:
        - 时段: 20%
        - 位置类型: 20%
        - 天气: 15%
        - 交通: 15%
        - 道路类型: 10%
        - 行程目的: 20%
        """
        score = 0.0
        total_weight = 0.0

        weights = {
            "time_of_day": 0.20,
            "location_type": 0.20,
            "weather": 0.15,
            "traffic_condition": 0.15,
            "road_type": 0.10,
            "trip_purpose": 0.20,
        }

        for attr, weight in weights.items():
            val_a = getattr(scene_a, attr, "")
            val_b = getattr(scene_b, attr, "")
            if val_a and val_b:
                total_weight += weight
                if val_a == val_b:
                    score += weight

        if total_weight == 0:
            return 0.0

        return score / total_weight

    async def get_context_aware_memories(
        self,
        user_id: str,
        current_scene: SceneContext,
        max_results: int = 10,
    ) -> MemoryRetrievalResult:
        """获取与当前场景相关的记忆"""
        # 查询该用户的所有场景记忆和情景记忆
        query = MemoryQuery(
            user_id=user_id,
            memory_types=[MemoryType.SCENE, MemoryType.EPISODIC],
            context=current_scene,
            max_results=max_results * 3,  # 多取一些用于后过滤
            sort_by="relevance",
        )

        result = await self._memory_store.query(query)

        # 计算场景相似度并排序
        scored_items = []
        for item in result.items:
            scene_similarity = 0.0
            if item.context_keys:
                # 解析上下文键并计算相似度
                for ctx_key in item.context_keys:
                    parsed_scene = self._parse_context_key(ctx_key)
                    if parsed_scene:
                        sim = self.calculate_context_similarity(
                            current_scene, parsed_scene
                        )
                        scene_similarity = max(scene_similarity, sim)

            scored_items.append((item, scene_similarity))

        # 按综合得分排序：场景相似度 * 0.5 + 记忆强度 * 0.3 + 重要性 * 0.2
        scored_items.sort(
            key=lambda x: (
                x[1] * 0.5
                + x[0].strength * 0.3
                + (x[0].importance.value / 5) * 0.2
            ),
            reverse=True,
        )

        result.items = [item for item, _ in scored_items[:max_results]]
        result.context_match_score = (
            scored_items[0][1] if scored_items else 0.0
        )
        return result

    async def predict_user_needs(
        self,
        user_id: str,
        current_scene: SceneContext,
        profile: Optional[UserProfile] = None,
    ) -> list[dict[str, Any]]:
        """基于当前场景预测用户需求

        返回预测的需求列表，按置信度排序
        """
        predictions: list[dict[str, Any]] = []

        # 1. 查询匹配的行为模式
        patterns = await self._pattern_store.get_patterns(user_id)
        for pattern in patterns:
            if self._pattern_matches_context(pattern, current_scene):
                predictions.append({
                    "type": "pattern_match",
                    "pattern_name": pattern.pattern_name,
                    "action": pattern.expected_action,
                    "confidence": pattern.confidence,
                    "source": "behavior_pattern",
                })

        # 2. 查询相似场景的历史记忆
        context_result = await self.get_context_aware_memories(
            user_id, current_scene, max_results=5
        )
        for item in context_result.items:
            if item.memory_type == MemoryType.EPISODIC:
                predictions.append({
                    "type": "historical_similar",
                    "intent": item.content.get("intent", ""),
                    "action": item.content.get("processed_input", {}),
                    "confidence": item.strength * 0.7,
                    "source": "episodic_memory",
                })

        # 3. 基于用户画像的默认推荐
        if profile:
            # 温度推荐
            if profile.temperature_preference:
                predictions.append({
                    "type": "profile_default",
                    "action": {"set_temperature": profile.temperature_preference},
                    "confidence": profile.confidence_score * 0.8,
                    "source": "user_profile",
                })

            # 驾驶模式推荐
            if profile.driving_mode_preference and current_scene.traffic_condition == "smooth":
                predictions.append({
                    "type": "profile_default",
                    "action": {"set_driving_mode": profile.driving_mode_preference},
                    "confidence": profile.confidence_score * 0.6,
                    "source": "user_profile",
                })

        # 按置信度排序
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:5]

    def detect_scene_change(
        self, previous: SceneContext, current: SceneContext
    ) -> dict[str, Any]:
        """检测场景变化并返回变化详情"""
        changes = {}

        if previous.engine_status != current.engine_status:
            changes["engine_status"] = {
                "from": previous.engine_status,
                "to": current.engine_status,
            }

        if previous.time_of_day != current.time_of_day:
            changes["time_of_day"] = {
                "from": previous.time_of_day,
                "to": current.time_of_day,
            }

        if previous.weather != current.weather:
            changes["weather"] = {
                "from": previous.weather,
                "to": current.weather,
            }

        if previous.traffic_condition != current.traffic_condition:
            changes["traffic"] = {
                "from": previous.traffic_condition,
                "to": current.traffic_condition,
            }

        if previous.location_type != current.location_type:
            changes["location"] = {
                "from": previous.location_type,
                "to": current.location_type,
            }

        if previous.passengers_count != current.passengers_count:
            changes["passengers"] = {
                "from": previous.passengers_count,
                "to": current.passengers_count,
            }

        similarity = self.calculate_context_similarity(previous, current)

        return {
            "has_changed": len(changes) > 0,
            "changes": changes,
            "similarity": similarity,
            "is_significant_change": similarity < 0.5,
        }

    def _pattern_matches_context(
        self, pattern: BehaviorPattern, context: SceneContext
    ) -> bool:
        """检查行为模式是否匹配当前场景"""
        triggers = pattern.trigger_conditions
        if not triggers:
            return True  # 无条件的模式始终匹配

        for key, expected_value in triggers.items():
            if key == "context_key":
                if context.get_context_key() != expected_value:
                    return False
            elif key == "time_of_day":
                if context.time_of_day != expected_value:
                    return False
            elif key == "day_of_week":
                if datetime.now().strftime("%A") != expected_value:
                    return False
            elif key == "weather":
                if context.weather != expected_value:
                    return False
            elif key == "traffic_condition":
                if context.traffic_condition != expected_value:
                    return False

        return True

    def _parse_context_key(self, key: str) -> Optional[SceneContext]:
        """解析上下文键为 SceneContext 对象"""
        parts = key.split("|")
        if len(parts) >= 5:
            return SceneContext(
                time_of_day=parts[0],
                location_type=parts[1],
                weather=parts[2],
                traffic_condition=parts[3],
                road_type=parts[4],
            )
        return None