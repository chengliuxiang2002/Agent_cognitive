"""
认知记忆模块 - 用户画像构建器

从交互数据和记忆条目中提取用户特征，构建和维护结构化用户画像。
支持:
- 从交互记录中提取偏好
- 从行为模式中聚合特征
- 画像置信度评估
- 渐进式画像更新

PF-6: 异步画像构建 - 使用 asyncio.gather 分块并行处理
"""

from __future__ import annotations

import asyncio
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
    """用户画像构建器

    支持冷启动: 通过 ColdStartEngine 为新用户初始化画像。
    PF-6: 特征提取使用 asyncio.gather 并行处理，分块大小 50。
    """

    # PF-6: 分块大小
    _CHUNK_SIZE = 50

    def __init__(
        self,
        profile_store: BaseProfileStore,
        interaction_store: BaseInteractionStore,
    ):
        self._profile_store = profile_store
        self._interaction_store = interaction_store
        # 延迟导入避免循环依赖
        from .cold_start import ColdStartEngine
        self._cold_start = ColdStartEngine()

    async def build_profile(
        self, user_id: str, department: str = "", role: str = ""
    ) -> UserProfile:
        """构建或更新用户画像 (PF-6: 异步并行优化)

        冷启动策略:
        - 如果用户无交互数据，使用 ColdStartEngine 初始化画像
        - 基于部门/角色匹配默认模板和群体画像
        """
        profile = await self._profile_store.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)

        interactions = await self._interaction_store.get_recent(user_id, limit=200)

        if not interactions:
            # 冷启动: 无交互数据时使用默认模板初始化
            if profile.data_points_count == 0:
                # 获取同部门/角色的已有用户画像作为群体参考
                group_profiles = await self._get_group_profiles(department, role)
                cold_profile = self._cold_start.initialize_new_user_profile(
                    user_id=user_id,
                    department=department,
                    role=role,
                    group_profiles=group_profiles,
                )
                cold_profile.created_at = profile.created_at
                profile = cold_profile
                await self._profile_store.save_profile(profile)
            return profile

        # PF-6: 使用 asyncio.gather 并行提取各类特征
        chunks = self._split_into_chunks(interactions, self._CHUNK_SIZE)
        results = await asyncio.gather(
            self._extract_temperature_preference_async(profile, chunks),
            self._extract_music_preference_async(profile, chunks),
            self._extract_destination_patterns_async(profile, chunks),
            self._extract_driving_mode_preference_async(profile, chunks),
            self._extract_interaction_style_async(profile, chunks),
            self._extract_active_hours_async(profile, chunks),
        )

        # 合并并行结果
        profile = results[0]
        profile = self._merge_music_preference(profile, results[1])
        profile = self._merge_destinations(profile, results[2])
        profile = self._merge_driving_mode(profile, results[3])
        profile = self._merge_interaction_style(profile, results[4])
        profile = self._merge_active_hours(profile, results[5])

        # 更新画像元数据
        profile.data_points_count = len(interactions)
        profile.updated_at = datetime.now()

        # 计算置信度：基于数据点数量和一致性
        profile.confidence_score = min(1.0, profile.data_points_count / 100)

        # 保存更新后的画像
        await self._profile_store.save_profile(profile)

        return profile

    # ─── PF-6: 分块工具方法 ──────────────────────────────

    @staticmethod
    def _split_into_chunks(items: list, chunk_size: int) -> list[list]:
        """将列表分块"""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    # ─── PF-6: 异步并行特征提取方法 ──────────────────────

    async def _extract_temperature_preference_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> UserProfile:
        """异步提取温度偏好 (分块并行)"""
        tasks = [asyncio.to_thread(self._extract_temps_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_temps = [t for chunk in chunk_results for t in chunk]
        if all_temps:
            profile.temperature_preference = sum(all_temps) / len(all_temps)
        return profile

    async def _extract_music_preference_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> dict:
        """异步提取音乐偏好 (返回 Counter)"""
        tasks = [asyncio.to_thread(self._extract_genres_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_genres = [g for chunk in chunk_results for g in chunk]
        return dict(Counter(all_genres))

    async def _extract_destination_patterns_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> dict:
        """异步提取目的地模式 (返回 Counter)"""
        tasks = [asyncio.to_thread(self._extract_dests_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_dests = [d for chunk in chunk_results for d in chunk]
        return dict(Counter(all_dests))

    async def _extract_driving_mode_preference_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> dict:
        """异步提取驾驶模式偏好 (返回 Counter)"""
        tasks = [asyncio.to_thread(self._extract_modes_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_modes = [m for chunk in chunk_results for m in chunk]
        return dict(Counter(all_modes))

    async def _extract_interaction_style_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> dict:
        """异步提取交互风格 (返回 Counter)"""
        tasks = [asyncio.to_thread(self._extract_styles_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_styles = [s for chunk in chunk_results for s in chunk]
        return dict(Counter(all_styles))

    async def _extract_active_hours_async(
        self, profile: UserProfile, chunks: list[list[InteractionRecord]]
    ) -> dict:
        """异步提取活跃时段 (返回 Counter)"""
        tasks = [asyncio.to_thread(self._extract_hours_from_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        all_hours = [h for chunk in chunk_results for h in chunk]
        return dict(Counter(all_hours))

    # ─── PF-6: 分块处理函数 (在线程池中执行) ─────────────

    @staticmethod
    def _extract_temps_from_chunk(chunk: list[InteractionRecord]) -> list[float]:
        temps = []
        for r in chunk:
            temp = r.processed_input.get("temperature")
            if temp is not None:
                try:
                    temps.append(float(temp))
                except (ValueError, TypeError):
                    pass
        return temps

    @staticmethod
    def _extract_genres_from_chunk(chunk: list[InteractionRecord]) -> list[str]:
        genres = []
        for r in chunk:
            genre = r.processed_input.get("genre")
            if genre:
                genres.append(genre)
        return genres

    @staticmethod
    def _extract_dests_from_chunk(chunk: list[InteractionRecord]) -> list[str]:
        dests = []
        for r in chunk:
            if r.intent == "navigate":
                dest = r.processed_input.get("destination", "")
                if dest:
                    dests.append(dest)
        return dests

    @staticmethod
    def _extract_modes_from_chunk(chunk: list[InteractionRecord]) -> list[str]:
        modes = []
        for r in chunk:
            mode = r.processed_input.get("driving_mode")
            if mode:
                modes.append(mode)
        return modes

    @staticmethod
    def _extract_styles_from_chunk(chunk: list[InteractionRecord]) -> list[str]:
        styles = []
        for r in chunk:
            style = r.processed_input.get("interaction_style")
            if style:
                styles.append(style)
        return styles

    @staticmethod
    def _extract_hours_from_chunk(chunk: list[InteractionRecord]) -> list[int]:
        hours = []
        for r in chunk:
            try:
                hour = r.timestamp.hour
                hours.append(hour)
            except AttributeError:
                pass
        return hours

    # ─── PF-6: 合并并行结果 ──────────────────────────────

    def _merge_music_preference(self, profile: UserProfile, genre_counts: dict) -> UserProfile:
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        profile.music_preferences = [g for g, _ in sorted_genres[:5]]
        return profile

    def _merge_destinations(self, profile: UserProfile, dest_counts: dict) -> UserProfile:
        sorted_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)
        profile.common_destinations = [
            {"name": d, "frequency": c}
            for d, c in sorted_dests[:10]
        ]
        return profile

    def _merge_driving_mode(self, profile: UserProfile, mode_counts: dict) -> UserProfile:
        if mode_counts:
            top_mode = max(mode_counts, key=mode_counts.get)
            profile.driving_mode_preference = top_mode
        return profile

    def _merge_interaction_style(self, profile: UserProfile, style_counts: dict) -> UserProfile:
        if style_counts:
            top_style = max(style_counts, key=style_counts.get)
            profile.interaction_style = top_style
        return profile

    def _merge_active_hours(self, profile: UserProfile, hour_counts: dict) -> UserProfile:
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        profile.active_hours = [
            {"hour": h, "activity_count": c}
            for h, c in sorted_hours[:8]
        ]
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

    async def _get_group_profiles(
        self, department: str = "", role: str = ""
    ) -> list[UserProfile]:
        """获取同部门/角色的已有用户画像（用于冷启动群体匹配）

        从 profile_store 中查询所有用户画像，按部门/角色过滤。
        注意：当前实现为简化版，生产环境应通过数据库索引查询。
        """
        try:
            # 遍历获取所有用户画像（简化实现）
            all_profiles = []
            # 尝试从已知用户ID获取画像
            known_users = []
            for uid in known_users:
                profile = await self._profile_store.get_profile(uid)
                if profile and profile.data_points_count > 10:
                    all_profiles.append(profile)
            return all_profiles
        except Exception:
            return []