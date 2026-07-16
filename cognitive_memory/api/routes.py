"""
认知记忆模块 - API 接口层

提供 RESTful API 接口供智能座舱其他模块调用。
接口设计原则:
- RESTful 风格
- JSON 数据格式
- 异步非阻塞
- 支持批量操作

注意: 生产环境使用 FastAPI，此处提供 API 路由定义和接口规范。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ─── 请求/响应模型 ────────────────────────────────────────

@dataclass
class RecordInteractionRequest:
    """记录交互请求"""
    user_id: str
    interaction_type: str = ""
    intent: str = ""
    raw_input: str = ""
    processed_input: dict[str, Any] = field(default_factory=dict)
    scene_context: Optional[dict[str, Any]] = None
    system_response: dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    user_satisfaction: Optional[float] = None
    was_successful: bool = True
    session_id: str = ""


@dataclass
class QueryMemoriesRequest:
    """查询记忆请求"""
    user_id: str
    memory_types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_results: int = 20
    sort_by: str = "relevance"
    time_range_days: Optional[int] = None
    scene_context: Optional[dict[str, Any]] = None


@dataclass
class PredictNeedsRequest:
    """预测需求请求"""
    user_id: str
    scene_context: dict[str, Any]


@dataclass
class ApiResponse:
    """统一API响应"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ─── API 路由定义 ────────────────────────────────────────

class MemoryAPI:
    """认知记忆模块 API

    接口规范:

    POST /api/v1/memory/interactions
        - 记录用户交互
        - Request: RecordInteractionRequest
        - Response: {"memory_id": "..."}

    POST /api/v1/memory/query
        - 查询记忆
        - Request: QueryMemoriesRequest
        - Response: MemoryRetrievalResult

    GET /api/v1/memory/profile/{user_id}
        - 获取用户画像
        - Response: UserProfile

    POST /api/v1/memory/profile/{user_id}/build
        - 构建/更新用户画像
        - Response: UserProfile

    POST /api/v1/memory/predict
        - 预测用户需求
        - Request: PredictNeedsRequest
        - Response: list[Prediction]

    GET /api/v1/memory/patterns/{user_id}
        - 获取行为模式
        - Query: ?pattern_type=xxx
        - Response: list[BehaviorPattern]

    DELETE /api/v1/memory/user/{user_id}
        - 删除用户所有数据（GDPR合规）
        - Response: {"deleted_count": N}

    GET /api/v1/memory/stats
        - 获取系统统计信息
        - Response: stats dict
    """

    def __init__(self, memory_manager):
        self._manager = memory_manager

    async def record_interaction(
        self, request: RecordInteractionRequest
    ) -> ApiResponse:
        """记录用户交互"""
        from ..models.memory import InteractionRecord, SceneContext

        scene = None
        if request.scene_context:
            scene = SceneContext(**request.scene_context)

        interaction = InteractionRecord(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            intent=request.intent,
            raw_input=request.raw_input,
            processed_input=request.processed_input,
            scene_context=scene,
            system_response=request.system_response,
            response_time_ms=request.response_time_ms,
            user_satisfaction=request.user_satisfaction,
            was_successful=request.was_successful,
            session_id=request.session_id,
        )

        try:
            memory_id = await self._manager.record_interaction(interaction)
            return ApiResponse(success=True, data={"memory_id": memory_id})
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def query_memories(
        self, request: QueryMemoriesRequest
    ) -> ApiResponse:
        """查询记忆"""
        from ..models.memory import MemoryType, SceneContext

        memory_types = None
        if request.memory_types:
            memory_types = [MemoryType(mt) for mt in request.memory_types]

        scene = None
        if request.scene_context:
            scene = SceneContext(**request.scene_context)

        try:
            result = await self._manager.query_memories(
                user_id=request.user_id,
                memory_types=memory_types,
                context=scene,
                tags=request.tags,
                keywords=request.keywords,
                max_results=request.max_results,
                sort_by=request.sort_by,
                time_range_days=request.time_range_days,
            )
            return ApiResponse(
                success=True,
                data={
                    "items": [item.to_dict() for item in result.items],
                    "total_found": result.total_found,
                    "retrieval_time_ms": result.retrieval_time_ms,
                },
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_profile(self, user_id: str) -> ApiResponse:
        """获取用户画像"""
        try:
            profile = await self._manager.get_user_profile(user_id)
            if profile:
                return ApiResponse(success=True, data=profile.to_dict())
            return ApiResponse(success=True, data=None)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def build_profile(self, user_id: str) -> ApiResponse:
        """构建用户画像"""
        try:
            profile = await self._manager.build_user_profile(user_id)
            return ApiResponse(success=True, data=profile.to_dict())
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def predict_needs(
        self, request: PredictNeedsRequest
    ) -> ApiResponse:
        """预测用户需求"""
        from ..models.memory import SceneContext

        scene = SceneContext(**request.scene_context)

        try:
            predictions = await self._manager.predict_user_needs(
                request.user_id, scene
            )
            return ApiResponse(success=True, data=predictions)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_patterns(
        self, user_id: str, pattern_type: Optional[str] = None
    ) -> ApiResponse:
        """获取行为模式"""
        try:
            patterns = await self._manager.get_behavior_patterns(
                user_id, pattern_type
            )
            return ApiResponse(
                success=True,
                data=[p.to_dict() for p in patterns],
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def forget_user(self, user_id: str) -> ApiResponse:
        """删除用户数据"""
        try:
            count = await self._manager.forget_user_data(user_id)
            return ApiResponse(
                success=True, data={"deleted_count": count}
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_stats(self) -> ApiResponse:
        """获取系统统计"""
        try:
            stats = await self._manager.get_stats()
            return ApiResponse(success=True, data=stats)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))