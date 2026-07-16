"""
认知记忆模块 - 行为模式学习器

从用户交互数据中自动学习行为模式，包括:
- 路线偏好学习
- 温度/空调偏好学习
- 音乐/媒体偏好学习
- 交互风格学习
- 时间模式学习

算法:
- 基于频率的统计学习
- 时间序列相关性分析
- 上下文关联规则挖掘
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from ..models.memory import (
    UserProfile,
    InteractionRecord,
    BehaviorPattern,
    SceneContext,
    MemoryItem,
    MemoryType,
    MemoryImportance,
)


class PatternLearner:
    """行为模式学习器 - 从交互数据中提取用户行为模式"""

    def __init__(self, min_occurrences: int = 3, confidence_threshold: float = 0.5):
        self._min_occurrences = min_occurrences
        self._confidence_threshold = confidence_threshold

    def learn_temperature_preference(
        self, interactions: list[InteractionRecord], profile: UserProfile
    ) -> list[BehaviorPattern]:
        """学习温度偏好模式"""
        patterns: list[BehaviorPattern] = []
        temp_settings: list[tuple[float, str]] = []  # (temperature, context_key)

        for record in interactions:
            if "temperature" in record.processed_input:
                ctx = record.scene_context
                temp = record.processed_input.get("temperature")
                if temp is not None and ctx:
                    context_key = ctx.get_context_key()
                    temp_settings.append((float(temp), context_key))

        if not temp_settings:
            return patterns

        # 按上下文分组统计
        context_groups: dict[str, list[float]] = defaultdict(list)
        for temp, ctx_key in temp_settings:
            context_groups[ctx_key].append(temp)

        for ctx_key, temps in context_groups.items():
            if len(temps) >= self._min_occurrences:
                avg_temp = sum(temps) / len(temps)
                variance = sum((t - avg_temp) ** 2 for t in temps) / len(temps)
                confidence = 1.0 / (1.0 + variance)

                if confidence >= self._confidence_threshold:
                    patterns.append(BehaviorPattern(
                        id=str(uuid.uuid4()),
                        user_id=profile.user_id,
                        pattern_name=f"temperature_pref_{ctx_key}",
                        pattern_type="temperature",
                        trigger_conditions={"context_key": ctx_key},
                        expected_action={"set_temperature": round(avg_temp, 1)},
                        occurrence_count=len(temps),
                        success_rate=confidence,
                        confidence=confidence,
                        first_observed=datetime.now(),
                        last_observed=datetime.now(),
                        related_context_keys=[ctx_key],
                    ))

        return patterns

    def learn_route_preference(
        self, interactions: list[InteractionRecord], profile: UserProfile
    ) -> list[BehaviorPattern]:
        """学习路线偏好模式"""
        patterns: list[BehaviorPattern] = []
        destinations: list[dict[str, Any]] = []

        for record in interactions:
            if record.intent == "navigate" and record.scene_context:
                dest = record.processed_input.get("destination", "")
                if dest:
                    destinations.append({
                        "destination": dest,
                        "time_of_day": record.scene_context.time_of_day,
                        "day_of_week": record.timestamp.strftime("%A"),
                        "trip_purpose": record.scene_context.trip_purpose,
                    })

        # 按目的地+时段分组
        time_slot_groups: dict[str, list[dict]] = defaultdict(list)
        for d in destinations:
            key = f"{d['destination']}|{d['time_of_day']}"
            time_slot_groups[key].append(d)

        for key, entries in time_slot_groups.items():
            if len(entries) >= self._min_occurrences:
                dest, time_slot = key.split("|")
                confidence = min(1.0, len(entries) / (self._min_occurrences * 2))

                patterns.append(BehaviorPattern(
                    id=str(uuid.uuid4()),
                    user_id=profile.user_id,
                    pattern_name=f"route_pref_{dest}_{time_slot}",
                    pattern_type="route",
                    trigger_conditions={
                        "time_of_day": time_slot,
                        "day_of_week": entries[0].get("day_of_week"),
                    },
                    expected_action={
                        "navigate_to": dest,
                        "trip_purpose": entries[0].get("trip_purpose", ""),
                    },
                    occurrence_count=len(entries),
                    success_rate=confidence,
                    confidence=confidence,
                    first_observed=datetime.now(),
                    last_observed=datetime.now(),
                    related_context_keys=[time_slot],
                ))

        return patterns

    def learn_media_preference(
        self, interactions: list[InteractionRecord], profile: UserProfile
    ) -> list[BehaviorPattern]:
        """学习媒体/音乐偏好模式"""
        patterns: list[BehaviorPattern] = []
        media_actions: list[dict[str, Any]] = []

        for record in interactions:
            if record.intent in ("play_music", "media_control"):
                media_actions.append({
                    "action": record.processed_input.get("action", ""),
                    "genre": record.processed_input.get("genre", ""),
                    "artist": record.processed_input.get("artist", ""),
                    "context_key": record.scene_context.get_context_key()
                    if record.scene_context else "",
                })

        # 统计音乐类型偏好
        genre_counter = Counter(
            a["genre"] for a in media_actions if a["genre"]
        )
        artist_counter = Counter(
            a["artist"] for a in media_actions if a["artist"]
        )

        total = sum(genre_counter.values())
        if total >= self._min_occurrences:
            for genre, count in genre_counter.most_common(5):
                confidence = count / total
                if confidence >= self._confidence_threshold:
                    patterns.append(BehaviorPattern(
                        id=str(uuid.uuid4()),
                        user_id=profile.user_id,
                        pattern_name=f"media_genre_{genre}",
                        pattern_type="media",
                        trigger_conditions={},
                        expected_action={"preferred_genre": genre},
                        occurrence_count=count,
                        success_rate=confidence,
                        confidence=confidence,
                        first_observed=datetime.now(),
                        last_observed=datetime.now(),
                    ))

        return patterns

    def learn_interaction_style(
        self, interactions: list[InteractionRecord], profile: UserProfile
    ) -> Optional[BehaviorPattern]:
        """学习用户交互风格"""
        if len(interactions) < self._min_occurrences:
            return None

        # 分析交互频率
        if len(interactions) < 2:
            return None

        time_span = (
            interactions[-1].timestamp - interactions[0].timestamp
        ).total_seconds()
        if time_span <= 0:
            return None

        freq = len(interactions) / (time_span / 3600.0)  # 每小时交互次数

        if freq > 10:
            style = "frequent"
        elif freq > 3:
            style = "moderate"
        else:
            style = "minimal"

        # 分析交互类型分布
        type_counter = Counter(r.interaction_type for r in interactions)
        preferred_type = type_counter.most_common(1)[0][0]

        # 分析满意度
        satisfaction_scores = [
            r.user_satisfaction for r in interactions
            if r.user_satisfaction is not None
        ]
        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores else 0.5
        )

        return BehaviorPattern(
            id=str(uuid.uuid4()),
            user_id=profile.user_id,
            pattern_name="interaction_style",
            pattern_type="interaction",
            trigger_conditions={},
            expected_action={
                "style": style,
                "preferred_input_type": preferred_type,
            },
            occurrence_count=len(interactions),
            success_rate=avg_satisfaction,
            confidence=min(1.0, len(interactions) / 20),
            first_observed=interactions[0].timestamp,
            last_observed=interactions[-1].timestamp,
        )

    def learn_time_patterns(
        self, interactions: list[InteractionRecord], profile: UserProfile
    ) -> list[BehaviorPattern]:
        """学习时间行为模式"""
        patterns: list[BehaviorPattern] = []

        # 按小时统计活跃度
        hour_activity = Counter(
            r.timestamp.hour for r in interactions
        )

        if len(interactions) >= self._min_occurrences:
            active_hours = [
                {"hour": h, "count": c}
                for h, c in hour_activity.most_common(8)
            ]

            patterns.append(BehaviorPattern(
                id=str(uuid.uuid4()),
                user_id=profile.user_id,
                pattern_name="active_hours",
                pattern_type="time",
                trigger_conditions={},
                expected_action={"active_hours": active_hours},
                occurrence_count=len(interactions),
                success_rate=1.0,
                confidence=min(1.0, len(interactions) / 30),
                first_observed=interactions[0].timestamp,
                last_observed=interactions[-1].timestamp,
            ))

        return patterns

    def learn_all(
        self,
        interactions: list[InteractionRecord],
        profile: UserProfile,
    ) -> list[BehaviorPattern]:
        """从所有交互数据中学习所有类型的模式"""
        all_patterns: list[BehaviorPattern] = []

        all_patterns.extend(self.learn_temperature_preference(interactions, profile))
        all_patterns.extend(self.learn_route_preference(interactions, profile))
        all_patterns.extend(self.learn_media_preference(interactions, profile))

        style = self.learn_interaction_style(interactions, profile)
        if style:
            all_patterns.append(style)

        all_patterns.extend(self.learn_time_patterns(interactions, profile))

        return all_patterns


class MemoryEncoder:
    """记忆编码器 - 将交互数据转换为结构化记忆条目"""

    def encode_interaction(
        self, record: InteractionRecord, user_id: str
    ) -> MemoryItem:
        """将交互记录编码为记忆条目"""
        importance = self._assess_importance(record)

        content = {
            "intent": record.intent,
            "processed_input": record.processed_input,
            "system_response": record.system_response,
            "was_successful": record.was_successful,
        }

        tags = self._extract_tags(record)
        context_keys = (
            [record.scene_context.get_context_key()]
            if record.scene_context else []
        )

        return MemoryItem(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance=importance,
            tags=tags,
            context_keys=context_keys,
            source=record.interaction_type,
            confidence=record.user_satisfaction or 0.8,
            metadata={
                "interaction_id": record.id,
                "session_id": record.session_id,
                "response_time_ms": record.response_time_ms,
            },
        )

    def encode_scene(self, scene: SceneContext, user_id: str) -> MemoryItem:
        """将场景上下文编码为场景记忆"""
        return MemoryItem(
            user_id=user_id,
            memory_type=MemoryType.SCENE,
            content=scene.to_dict(),
            importance=MemoryImportance.LOW,
            tags=[scene.time_of_day, scene.location_type, scene.weather],
            context_keys=[scene.get_context_key()],
            source="scene_sensor",
        )

    def _assess_importance(self, record: InteractionRecord) -> MemoryImportance:
        """评估交互记录的重要性"""
        # 安全相关的交互 → CRITICAL
        safety_keywords = ["安全", "危险", "碰撞", "刹车", "安全气囊"]
        if any(kw in record.raw_input for kw in safety_keywords):
            return MemoryImportance.CRITICAL

        # 导航/路线相关 → HIGH
        if record.intent in ("navigate", "set_destination"):
            return MemoryImportance.HIGH

        # 偏好设置相关 → HIGH
        if record.intent in ("set_temperature", "adjust_seat", "set_preference"):
            return MemoryImportance.HIGH

        # 成功且有反馈的交互 → MEDIUM
        if record.was_successful and record.user_satisfaction is not None:
            return MemoryImportance.MEDIUM

        # 默认 → LOW
        return MemoryImportance.LOW

    def _extract_tags(self, record: InteractionRecord) -> list[str]:
        """从交互记录中提取标签"""
        tags = [record.interaction_type, record.intent]

        if record.scene_context:
            tags.extend([
                record.scene_context.time_of_day,
                record.scene_context.weather,
                record.scene_context.traffic_condition,
            ])

        if record.was_successful:
            tags.append("successful")
        else:
            tags.append("failed")

        return [t for t in tags if t]