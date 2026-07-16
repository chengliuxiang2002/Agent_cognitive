"""
认知记忆模块 - 数据模型定义

定义记忆系统中所有核心数据结构，包括:
- 记忆类型枚举
- 记忆条目
- 用户画像
- 交互记录
- 场景上下文
- 行为模式
- 记忆查询与检索结果
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"        # 短期记忆：当前会话上下文
    LONG_TERM = "long_term"          # 长期记忆：持久化用户偏好
    EPISODIC = "episodic"            # 情景记忆：具体事件/经历
    SEMANTIC = "semantic"            # 语义记忆：事实性知识
    PROCEDURAL = "procedural"        # 程序性记忆：操作习惯/流程
    SCENE = "scene"                  # 场景记忆：驾驶场景相关


class MemoryImportance(Enum):
    """记忆重要性等级"""
    CRITICAL = 5    # 关键信息（如安全偏好）
    HIGH = 4        # 高重要性
    MEDIUM = 3      # 中等重要性
    LOW = 2         # 低重要性
    TRANSIENT = 1   # 临时信息


@dataclass
class MemoryItem:
    """记忆条目 - 记忆系统的基本存储单元"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: dict[str, Any] = field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.MEDIUM

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    last_updated_at: datetime = field(default_factory=datetime.now)

    # 记忆强度 (0.0 ~ 1.0)，受衰减和强化影响
    strength: float = 1.0

    # 衰减相关
    decay_rate: float = 0.01         # 衰减速率
    min_strength: float = 0.1        # 最低强度阈值（低于此值可能被遗忘）

    # 关联标签和上下文
    tags: list[str] = field(default_factory=list)
    context_keys: list[str] = field(default_factory=list)  # 关联的场景上下文键

    # 访问次数（用于强化计算）
    access_count: int = 0

    # 元数据
    source: str = ""                 # 数据来源（如 "voice_interaction", "touch_input"）
    confidence: float = 1.0          # 置信度
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "importance": self.importance.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "strength": self.strength,
            "decay_rate": self.decay_rate,
            "tags": self.tags,
            "context_keys": self.context_keys,
            "access_count": self.access_count,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class UserProfile:
    """用户画像 - 用户特征的结构化档案"""
    user_id: str
    # 基本属性
    name: str = ""
    age_group: str = ""              # 年龄段: "18-25", "26-35", "36-50", "50+"
    gender: str = ""

    # 偏好设置
    temperature_preference: float = 23.0       # 温度偏好 (℃)
    seat_position_preference: dict[str, Any] = field(default_factory=dict)
    music_preferences: list[str] = field(default_factory=list)  # 音乐偏好
    driving_mode_preference: str = "comfort"  # 驾驶模式偏好: "comfort", "sport", "eco"
    voice_assistant_style: str = "concise"   # 语音助手风格: "concise", "detailed", "humorous"

    # 行为特征
    common_destinations: list[dict[str, Any]] = field(default_factory=list)
    frequent_routes: list[dict[str, Any]] = field(default_factory=list)
    driving_habits: list[str] = field(default_factory=list)
    interaction_style: str = "moderate"  # 交互风格: "minimal", "moderate", "frequent"

    # 健康与安全
    health_conditions: list[str] = field(default_factory=list)
    safety_preferences: dict[str, Any] = field(default_factory=dict)

    # 时间模式
    active_hours: list[dict[str, Any]] = field(default_factory=list)  # 活跃时段
    weekly_pattern: dict[str, Any] = field(default_factory=dict)      # 每周出行模式

    # 系统元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5     # 画像置信度
    data_points_count: int = 0        # 用于构建画像的数据点数量

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "age_group": self.age_group,
            "gender": self.gender,
            "temperature_preference": self.temperature_preference,
            "seat_position_preference": self.seat_position_preference,
            "music_preferences": self.music_preferences,
            "driving_mode_preference": self.driving_mode_preference,
            "voice_assistant_style": self.voice_assistant_style,
            "common_destinations": self.common_destinations,
            "frequent_routes": self.frequent_routes,
            "driving_habits": self.driving_habits,
            "interaction_style": self.interaction_style,
            "health_conditions": self.health_conditions,
            "safety_preferences": self.safety_preferences,
            "active_hours": self.active_hours,
            "weekly_pattern": self.weekly_pattern,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence_score": self.confidence_score,
            "data_points_count": self.data_points_count,
        }


@dataclass
class InteractionRecord:
    """用户交互记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # 交互类型
    interaction_type: str = ""       # "voice_command", "touch_input", "gesture", "automatic"
    intent: str = ""                 # 识别的意图
    raw_input: str = ""              # 原始输入
    processed_input: dict[str, Any] = field(default_factory=dict)

    # 上下文
    scene_context: Optional[SceneContext] = None

    # 系统响应
    system_response: dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0

    # 用户反馈
    user_satisfaction: Optional[float] = None  # 0.0 ~ 1.0
    was_successful: bool = True

    # 元数据
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "interaction_type": self.interaction_type,
            "intent": self.intent,
            "raw_input": self.raw_input,
            "processed_input": self.processed_input,
            "scene_context": self.scene_context.to_dict() if self.scene_context else None,
            "system_response": self.system_response,
            "response_time_ms": self.response_time_ms,
            "user_satisfaction": self.user_satisfaction,
            "was_successful": self.was_successful,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class SceneContext:
    """场景上下文 - 当前驾驶场景的完整描述"""
    # 时间
    timestamp: datetime = field(default_factory=datetime.now)
    time_of_day: str = ""            # "morning", "afternoon", "evening", "night"

    # 位置
    latitude: float = 0.0
    longitude: float = 0.0
    location_name: str = ""          # 语义化位置名称
    location_type: str = ""          # "home", "work", "mall", "restaurant", etc.

    # 车辆状态
    vehicle_speed: float = 0.0       # km/h
    engine_status: str = "off"       # "off", "idle", "driving"
    fuel_level: float = 0.0          # 油量/电量百分比
    odometer: float = 0.0            # 里程

    # 环境
    weather: str = ""                # "sunny", "rainy", "cloudy", "snowy"
    temperature_outside: float = 0.0
    traffic_condition: str = ""      # "smooth", "moderate", "heavy", "jammed"
    road_type: str = ""              # "highway", "urban", "rural", "parking"

    # 舱内状态
    cabin_temperature: float = 0.0
    passengers_count: int = 0
    music_playing: bool = False
    current_media: str = ""

    # 行程信息
    is_navigating: bool = False
    destination: str = ""
    estimated_arrival: Optional[datetime] = None
    trip_purpose: str = ""           # "commute", "leisure", "shopping", "school_run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_of_day": self.time_of_day,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_name": self.location_name,
            "location_type": self.location_type,
            "vehicle_speed": self.vehicle_speed,
            "engine_status": self.engine_status,
            "fuel_level": self.fuel_level,
            "odometer": self.odometer,
            "weather": self.weather,
            "temperature_outside": self.temperature_outside,
            "traffic_condition": self.traffic_condition,
            "road_type": self.road_type,
            "cabin_temperature": self.cabin_temperature,
            "passengers_count": self.passengers_count,
            "music_playing": self.music_playing,
            "current_media": self.current_media,
            "is_navigating": self.is_navigating,
            "destination": self.destination,
            "estimated_arrival": self.estimated_arrival.isoformat() if self.estimated_arrival else None,
            "trip_purpose": self.trip_purpose,
        }

    def get_context_key(self) -> str:
        """生成场景上下文的唯一标识键"""
        parts = [
            self.time_of_day,
            self.location_type or "unknown",
            self.weather or "unknown",
            self.traffic_condition or "unknown",
            self.road_type or "unknown",
        ]
        return "|".join(parts)


@dataclass
class BehaviorPattern:
    """用户行为模式 - 从历史数据中学习到的模式"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    pattern_name: str = ""
    pattern_type: str = ""           # "route", "temperature", "media", "interaction", etc.

    # 触发条件
    trigger_conditions: dict[str, Any] = field(default_factory=dict)

    # 模式行为
    expected_action: dict[str, Any] = field(default_factory=dict)

    # 统计数据
    occurrence_count: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0

    # 时间范围
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)

    # 关联上下文
    related_context_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "trigger_conditions": self.trigger_conditions,
            "expected_action": self.expected_action,
            "occurrence_count": self.occurrence_count,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "related_context_keys": self.related_context_keys,
        }


@dataclass
class MemoryQuery:
    """记忆查询请求"""
    user_id: str
    memory_types: list[MemoryType] = field(default_factory=list)
    context: Optional[SceneContext] = None
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    min_strength: float = 0.1
    min_importance: MemoryImportance = MemoryImportance.TRANSIENT
    max_results: int = 20
    time_range_days: Optional[int] = None
    sort_by: str = "relevance"  # "relevance", "recency", "strength", "importance"


@dataclass
class MemoryRetrievalResult:
    """记忆检索结果"""
    query: MemoryQuery
    items: list[MemoryItem] = field(default_factory=list)
    total_found: int = 0
    retrieval_time_ms: float = 0.0
    context_match_score: float = 0.0