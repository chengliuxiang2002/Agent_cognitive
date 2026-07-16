"""
认知记忆模块 - 用户画像构建器

从交互数据和记忆条目中提取用户特征，构建和维护结构化用户画像。
支持:
- 从交互记录中提取偏好
- 从行为模式中聚合特征
- 画像置信度评估
- 渐进式画像更新
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from ..models.memory import (
    UserProfile,
    InteractionRecord,
    BehaviorPattern,
    MemoryItem,
    MemoryType,
)
from ..storage.base import BaseInteractionStore, BaseProfileStore


class UserProfileBuilder:
    """用户画像构建器"""

    def __init__(
        self,
        profile_store: BaseProfileStore,
        interaction_store: BaseInteractionStore,
    ):
        self._profile_store = profile_store
        self._interaction_store = interaction_store

    async def build_profile(self, user_id: str) -> UserProfile:
        """构建或更新用户画像"""
        profile = await self._profile_store.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)

        interactions = await self._interaction_store.get_recent(user_id, limit=200)

        if not interactions:
            return profile

        # 提取各类特征
        profile = self._extract_temperature_preference(profile, interactions)
        profile = self._extract_music_preference(profile, interactions)
        profile = self._extract_destination_patterns(profile, interactions)
        profile = self._extract_driving_mode_preference(profile, interactions)
        profile = self._extract_interaction_style(profile, interactions)
        profile = self._extract_active_hours(profile, interactions)

        # 更新画像元数据
        profile.data_points_count = len(interactions)
        profile.updated_at = datetime.now()

        # 计算置信度：基于数据点数量和一致性
        profile.confidence_score = min(1.0, profile.data_points_count / 100)

        # 保存更新后的画像
        await self._profile_store.save_profile(profile)

        return profile

    async def update_from_patterns(
        self, user_id: str, patterns: list[BehaviorPattern]
    ) -> Optional[UserProfile]:
        """从行为模式更新用户画像"""
        profile = await self._profile_store.get_profile(user_id)
        if profile is None:
            return None

        for pattern in patterns:
            if pattern.confidence < 0.5:
                continue

            if pattern.pattern_type == "temperature":
                temp = pattern.expected_action.get("set_temperature")
                if temp is not None:
                    profile.temperature_preference = (
                        profile.temperature_preference * 0.7 + temp * 0.3
                    )

            elif pattern.pattern_type == "route":
                dest = pattern.expected_action.get("navigate_to", "")
                purpose = pattern.expected_action.get("trip_purpose", "")
                if dest:
                    existing = [
                        d for d in profile.common_destinations
                        if d.get("name") == dest
                    ]
                    if existing:
                        existing[0]["frequency"] = existing[0].get("frequency", 0) + 1
                    else:
                        profile.common_destinations.append({
                            "name": dest,
                            "purpose": purpose,
                            "frequency": 1,
                        })

            elif pattern.pattern_type == "media":
                genre = pattern.expected_action.get("preferred_genre")
                if genre and genre not in profile.music_preferences:
                    profile.music_preferences.append(genre)

            elif pattern.pattern_type == "interaction":
                style = pattern.expected_action.get("style")
                if style:
                    profile.interaction_style = style

            elif pattern.pattern_type == "time":
                active_hours = pattern.expected_action.get("active_hours", [])
                profile.active_hours = active_hours

        profile.updated_at = datetime.now()
        profile.confidence_score = min(1.0, profile.confidence_score + 0.05)
        await self._profile_store.save_profile(profile)

        return profile

    def _extract_temperature_preference(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        temps = []
        for r in interactions:
            temp = r.processed_input.get("temperature")
            if temp is not None:
                try:
                    temps.append(float(temp))
                except (ValueError, TypeError):
                    pass

        if temps:
            profile.temperature_preference = sum(temps) / len(temps)
        return profile

    def _extract_music_preference(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        genres = []
        for r in interactions:
            genre = r.processed_input.get("genre")
            if genre:
                genres.append(genre)

        genre_counts = Counter(genres)
        profile.music_preferences = [
            g for g, _ in genre_counts.most_common(5)
        ]
        return profile

    def _extract_destination_patterns(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        destinations = []
        for r in interactions:
            if r.intent == "navigate":
                dest = r.processed_input.get("destination", "")
                if dest:
                    destinations.append(dest)

        dest_counts = Counter(destinations)
        profile.common_destinations = [
            {"name": d, "frequency": c}
            for d, c in dest_counts.most_common(10)
        ]
        return profile

    def _extract_driving_mode_preference(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        modes = []
        for r in interactions:
            mode = r.processed_input.get("driving_mode")
            if mode:
                modes.append(mode)

        if modes:
            mode_counts = Counter(modes)
            profile.driving_mode_preference = mode_counts.most_common(1)[0][0]
        return profile

    def _extract_interaction_style(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        if len(interactions) < 3:
            return profile

        # 按交互类型分析
        type_counts = Counter(r.interaction_type for r in interactions)
        voice_ratio = type_counts.get("voice_command", 0) / len(interactions)

        if voice_ratio > 0.7:
            profile.interaction_style = "voice_first"
        elif voice_ratio > 0.3:
            profile.interaction_style = "moderate"
        else:
            profile.interaction_style = "touch_first"

        return profile

    def _extract_active_hours(
        self, profile: UserProfile, interactions: list[InteractionRecord]
    ) -> UserProfile:
        hour_counts = Counter(r.timestamp.hour for r in interactions)
        profile.active_hours = [
            {"hour": h, "activity_count": c}
            for h, c in hour_counts.most_common(8)
        ]
        return profile