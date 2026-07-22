"""
认知记忆模块 - 冷启动模块

解决新用户冷启动问题:
- 部门/角色相似性算法
- 群体画像匹配逻辑
- 新用户画像初始化
- 默认画像模板库（按部门/角色分类）
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from ..models.memory import UserProfile


# ─── 默认画像模板库 ──────────────────────────────────────

DEFAULT_PROFILE_TEMPLATES: dict[str, dict[str, Any]] = {
    "engineering": {
        "department": "engineering",
        "role": "engineer",
        "temperature_preference": 23.5,
        "driving_mode_preference": "comfort",
        "music_preferences": ["electronic", "ambient", "classical"],
        "voice_assistant_style": "concise",
        "interaction_style": "moderate",
        "description": "工程师：偏好简洁交互、舒适驾驶、专注型音乐",
    },
    "product": {
        "department": "product",
        "role": "product_manager",
        "temperature_preference": 24.0,
        "driving_mode_preference": "comfort",
        "music_preferences": ["pop", "jazz", "rock"],
        "voice_assistant_style": "detailed",
        "interaction_style": "frequent",
        "description": "产品经理：偏好详细交互、高频使用、多样化音乐",
    },
    "design": {
        "department": "design",
        "role": "designer",
        "temperature_preference": 23.0,
        "driving_mode_preference": "comfort",
        "music_preferences": ["jazz", "ambient", "indie"],
        "voice_assistant_style": "detailed",
        "interaction_style": "moderate",
        "description": "设计师：偏好详细交互、氛围型音乐、舒适驾驶",
    },
    "sales": {
        "department": "sales",
        "role": "sales",
        "temperature_preference": 22.5,
        "driving_mode_preference": "eco",
        "music_preferences": ["pop", "rock", "electronic"],
        "voice_assistant_style": "concise",
        "interaction_style": "frequent",
        "description": "销售：偏好简洁交互、高频使用、节能模式",
    },
    "hr": {
        "department": "hr",
        "role": "hr",
        "temperature_preference": 24.0,
        "driving_mode_preference": "comfort",
        "music_preferences": ["pop", "jazz", "classical"],
        "voice_assistant_style": "detailed",
        "interaction_style": "moderate",
        "description": "人力资源：偏好详细交互、舒适驾驶、轻松音乐",
    },
    "finance": {
        "department": "finance",
        "role": "finance",
        "temperature_preference": 23.5,
        "driving_mode_preference": "eco",
        "music_preferences": ["classical", "ambient", "jazz"],
        "voice_assistant_style": "concise",
        "interaction_style": "minimal",
        "description": "财务：偏好极简交互、节能模式、安静音乐",
    },
    "management": {
        "department": "management",
        "role": "manager",
        "temperature_preference": 23.5,
        "driving_mode_preference": "comfort",
        "music_preferences": ["classical", "jazz", "pop"],
        "voice_assistant_style": "concise",
        "interaction_style": "moderate",
        "description": "管理层：偏好简洁交互、舒适驾驶、经典音乐",
    },
    "default": {
        "department": "unknown",
        "role": "unknown",
        "temperature_preference": 23.0,
        "driving_mode_preference": "comfort",
        "music_preferences": ["pop", "rock", "electronic"],
        "voice_assistant_style": "concise",
        "interaction_style": "moderate",
        "description": "默认模板：通用画像配置",
    },
}


class ColdStartEngine:
    """冷启动引擎 - 解决新用户画像初始化问题

    核心策略:
    1. 基于部门/角色的默认模板匹配
    2. 群体画像相似度计算（找同部门/同角色已有用户）
    3. 渐进式画像融合（默认模板 + 群体画像 + 个人数据）
    """

    def __init__(
        self,
        templates: Optional[dict[str, dict[str, Any]]] = None,
    ):
        self._templates = templates or DEFAULT_PROFILE_TEMPLATES

    def get_default_template(self, department: str) -> dict[str, Any]:
        """获取部门默认画像模板"""
        return self._templates.get(
            department, self._templates["default"]
        )

    def calculate_group_similarity(
        self,
        target_profile: UserProfile,
        group_profiles: list[UserProfile],
    ) -> float:
        """计算目标用户与群体画像的相似度

        使用加权余弦相似度，考虑:
        - 温度偏好: 25%
        - 驾驶模式: 20%
        - 音乐偏好: 20%
        - 交互风格: 15%
        - 语音助手风格: 10%
        - 活跃时段: 10%
        """
        if not group_profiles:
            return 0.0

        similarities = []
        for profile in group_profiles:
            sim = self._calculate_profile_similarity(target_profile, profile)
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_profile_similarity(
        self, profile_a: UserProfile, profile_b: UserProfile
    ) -> float:
        """计算两个用户画像的相似度"""
        score = 0.0
        total_weight = 0.0

        weights = {
            "temperature": 0.25,
            "driving_mode": 0.20,
            "music": 0.20,
            "interaction": 0.15,
            "voice_style": 0.10,
            "active_hours": 0.10,
        }

        # 温度偏好相似度 (基于偏差)
        if profile_a.temperature_preference and profile_b.temperature_preference:
            temp_diff = abs(
                profile_a.temperature_preference - profile_b.temperature_preference
            )
            temp_sim = max(0.0, 1.0 - temp_diff / 10.0)  # 偏差10°C内线性衰减
            score += weights["temperature"] * temp_sim
            total_weight += weights["temperature"]

        # 驾驶模式
        if profile_a.driving_mode_preference and profile_b.driving_mode_preference:
            if profile_a.driving_mode_preference == profile_b.driving_mode_preference:
                score += weights["driving_mode"]
            total_weight += weights["driving_mode"]

        # 音乐偏好 (Jaccard相似度)
        if profile_a.music_preferences and profile_b.music_preferences:
            set_a = set(profile_a.music_preferences)
            set_b = set(profile_b.music_preferences)
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                score += weights["music"] * jaccard
            total_weight += weights["music"]

        # 交互风格
        if profile_a.interaction_style and profile_b.interaction_style:
            if profile_a.interaction_style == profile_b.interaction_style:
                score += weights["interaction"]
            total_weight += weights["interaction"]

        # 语音助手风格
        if profile_a.voice_assistant_style and profile_b.voice_assistant_style:
            if profile_a.voice_assistant_style == profile_b.voice_assistant_style:
                score += weights["voice_style"]
            total_weight += weights["voice_style"]

        return score / total_weight if total_weight > 0 else 0.0

    def initialize_new_user_profile(
        self,
        user_id: str,
        department: str = "",
        role: str = "",
        name: str = "",
        group_profiles: Optional[list[UserProfile]] = None,
    ) -> UserProfile:
        """为新用户初始化画像

        策略:
        1. 优先使用部门默认模板
        2. 如果有同部门/角色已有用户，融合群体画像特征
        3. 设置低置信度，等待数据积累后更新
        """
        template = self.get_default_template(department)

        profile = UserProfile(
            user_id=user_id,
            name=name,
            temperature_preference=template.get("temperature_preference", 23.0),
            driving_mode_preference=template.get("driving_mode_preference", "comfort"),
            music_preferences=template.get("music_preferences", []),
            voice_assistant_style=template.get("voice_assistant_style", "concise"),
            interaction_style=template.get("interaction_style", "moderate"),
            confidence_score=0.15,  # 冷启动低置信度
            data_points_count=0,
        )

        # 融合群体画像
        if group_profiles:
            profile = self._blend_with_group(profile, group_profiles, department)

        return profile

    def _blend_with_group(
        self,
        profile: UserProfile,
        group_profiles: list[UserProfile],
        department: str,
    ) -> UserProfile:
        """将个人默认画像与群体画像融合

        融合权重: 默认模板 60% + 群体画像 40%
        """
        if not group_profiles:
            return profile

        # 计算群体均值
        temps = [
            p.temperature_preference
            for p in group_profiles
            if p.temperature_preference
        ]
        if temps:
            group_temp = sum(temps) / len(temps)
            profile.temperature_preference = (
                profile.temperature_preference * 0.6 + group_temp * 0.4
            )

        # 群体驾驶模式众数
        modes = [p.driving_mode_preference for p in group_profiles if p.driving_mode_preference]
        if modes:
            mode_counts = Counter(modes)
            group_mode = mode_counts.most_common(1)[0][0]
            if group_mode != profile.driving_mode_preference and mode_counts[group_mode] > len(group_profiles) * 0.5:
                profile.driving_mode_preference = group_mode

        # 群体音乐偏好聚合
        all_genres = []
        for p in group_profiles:
            all_genres.extend(p.music_preferences)
        genre_counts = Counter(all_genres)
        top_genres = [g for g, _ in genre_counts.most_common(5)]
        existing = set(profile.music_preferences)
        for g in top_genres:
            if g not in existing and len(profile.music_preferences) < 5:
                profile.music_preferences.append(g)

        return profile

    def find_similar_users(
        self,
        profile: UserProfile,
        all_profiles: list[UserProfile],
        min_similarity: float = 0.6,
        top_k: int = 5,
    ) -> list[tuple[UserProfile, float]]:
        """查找与目标用户最相似的已有用户"""
        scored = []
        for p in all_profiles:
            if p.user_id == profile.user_id:
                continue
            sim = self._calculate_profile_similarity(profile, p)
            if sim >= min_similarity:
                scored.append((p, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def predict_user_preference(
        self,
        user_id: str,
        department: str,
        attribute: str,
        all_profiles: list[UserProfile],
        group_profiles: list[UserProfile],
    ) -> Any:
        """为冷启动用户预测特定偏好属性

        用于在用户数据不足时，快速给出合理默认值。
        """
        if group_profiles:
            if attribute == "temperature_preference":
                temps = [p.temperature_preference for p in group_profiles if p.temperature_preference]
                return sum(temps) / len(temps) if temps else 23.0
            elif attribute == "driving_mode_preference":
                modes = [p.driving_mode_preference for p in group_profiles if p.driving_mode_preference]
                return Counter(modes).most_common(1)[0][0] if modes else "comfort"
            elif attribute == "music_preferences":
                all_genres = []
                for p in group_profiles:
                    all_genres.extend(p.music_preferences)
                return [g for g, _ in Counter(all_genres).most_common(5)]
            elif attribute == "interaction_style":
                styles = [p.interaction_style for p in group_profiles if p.interaction_style]
                return Counter(styles).most_common(1)[0][0] if styles else "moderate"

        # 回退到默认模板
        template = self.get_default_template(department)
        return template.get(attribute)

    def get_personalization_progress(
        self, profile: UserProfile, group_profiles: list[UserProfile]
    ) -> dict[str, Any]:
        """P2: 渐进式个性化追踪 — 衡量从协同到个人的过渡进度

        返回:
        - stage: "cold_start" | "hybrid" | "personalized"
        - personal_weight: 个人数据权重 (0~1)
        - group_weight: 群体数据权重 (0~1)
        - data_sufficiency: 数据充足度 (0~1)
        """
        data_points = profile.data_points_count
        confidence = profile.confidence_score

        # 阶段判定
        if data_points < 10 or confidence < 0.3:
            stage = "cold_start"
            personal_weight = max(0.1, min(0.4, data_points / 10.0))
        elif data_points < 30 or confidence < 0.6:
            stage = "hybrid"
            personal_weight = 0.4 + 0.3 * min(1.0, (data_points - 10) / 20.0)
        else:
            stage = "personalized"
            personal_weight = 0.7 + 0.3 * min(1.0, (data_points - 30) / 50.0)

        group_weight = 1.0 - personal_weight
        data_sufficiency = min(1.0, data_points / 50.0)

        return {
            "stage": stage,
            "personal_weight": round(personal_weight, 3),
            "group_weight": round(group_weight, 3),
            "data_sufficiency": round(data_sufficiency, 3),
            "data_points": data_points,
            "confidence": round(confidence, 3),
        }