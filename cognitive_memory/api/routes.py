"""
认知记忆模块 - API 接口层

提供 RESTful API 接口供智能座舱其他模块调用。
接口设计原则:
- RESTful 风格
- JSON 数据格式
- 异步非阻塞
- 支持批量操作

SC-6: API 认证集成 - Token 验证中间件、SSO/OAuth2.0 集成

注意: 生产环境使用 FastAPI，此处提供 API 路由定义和接口规范。
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SC-6: API 认证机制
# ═══════════════════════════════════════════════════════════════════════════════


class UnauthorizedError(Exception):
    """未认证异常"""
    pass


class TokenExpiredError(Exception):
    """Token 过期异常"""
    pass


class AuthToken:
    """SC-6: 认证 Token 管理

    支持:
    - Token 生成与验证 (HMAC-SHA256)
    - Token 过期管理
    - 从 Token 中提取 user_id
    """

    def __init__(self, secret_key: Optional[bytes] = None):
        import os
        self._secret_key = secret_key or os.urandom(32)
        self._token_duration_s = 3600  # 默认 1 小时
        self._refresh_window_s = 300   # 过期前 5 分钟可刷新

    def generate_token(self, user_id: str, extra_claims: Optional[dict[str, Any]] = None) -> str:
        """生成认证 Token

        Token 格式: base64(user_id:expiry:hmac_signature)
        """
        expiry = int(time.time()) + self._token_duration_s
        claims = {
            "user_id": user_id,
            "exp": expiry,
            "iat": int(time.time()),
        }
        if extra_claims:
            claims.update(extra_claims)

        payload = f"{claims['user_id']}:{claims['exp']}:{claims['iat']}"
        signature = self._sign(payload)

        import base64
        token_data = json.dumps(claims, ensure_ascii=False)
        return base64.urlsafe_b64encode(token_data.encode()).decode() + "." + signature

    def validate_token(self, token: str) -> dict[str, Any]:
        """验证 Token 并返回 claims

        Raises:
            UnauthorizedError: Token 无效
            TokenExpiredError: Token 已过期
        """
        import base64

        try:
            # 分离 token 数据与签名
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                raise UnauthorizedError("Invalid token format")

            token_data_b64, signature = parts
            token_data = json.loads(base64.urlsafe_b64decode(token_data_b64).decode())

            # 验证签名
            payload = f"{token_data['user_id']}:{token_data['exp']}:{token_data['iat']}"
            expected_sig = self._sign(payload)
            if not hmac.compare_digest(signature, expected_sig):
                raise UnauthorizedError("Invalid token signature")

            # 验证过期
            if token_data["exp"] < time.time():
                raise TokenExpiredError("Token expired")

            return token_data

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise UnauthorizedError(f"Invalid token: {e}")

    def extract_user_id(self, token: str) -> str:
        """从 Token 中提取 user_id"""
        claims = self.validate_token(token)
        return claims["user_id"]

    def should_refresh(self, token: str) -> bool:
        """判断 Token 是否需要刷新 (过期前5分钟内)"""
        import base64
        try:
            parts = token.rsplit(".", 1)
            token_data = json.loads(base64.urlsafe_b64decode(parts[0]).decode())
            remaining = token_data["exp"] - time.time()
            return 0 < remaining < self._refresh_window_s
        except Exception:
            return False

    def refresh_token(self, token: str) -> str:
        """刷新 Token (返回新 Token)"""
        claims = self.validate_token(token)
        return self.generate_token(
            claims["user_id"],
            extra_claims={k: v for k, v in claims.items() if k not in ("user_id", "exp", "iat")},
        )

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret_key, payload.encode(), hashlib.sha256).hexdigest()


class SSOAuthService:
    """SC-6: SSO/OAuth2.0 认证服务集成

    对接公司统一身份认证系统:
    - 验证 OAuth2.0 Access Token
    - 从 SSO 服务获取 user_id
    - 支持 Token 缓存以减少认证请求
    """

    def __init__(self, sso_endpoint: str = "", client_id: str = "", client_secret: str = ""):
        self._sso_endpoint = sso_endpoint
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 分钟缓存

    async def verify_sso_token(self, access_token: str) -> Optional[str]:
        """通过 SSO 服务验证 Token 并返回 user_id

        Returns:
            user_id 或 None (验证失败)
        """
        # 缓存检查
        if access_token in self._token_cache:
            cached = self._token_cache[access_token]
            if cached["exp"] > time.time():
                return cached["user_id"]
            del self._token_cache[access_token]

        if not self._sso_endpoint:
            # 未配置 SSO 时，使用本地 Token 验证
            return None

        try:
            import urllib.request
            import urllib.error

            data = json.dumps({"token": access_token}).encode("utf-8")
            req = urllib.request.Request(
                f"{self._sso_endpoint}/oauth2/introspect",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._client_secret}",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())

            if result.get("active", False):
                user_id = result.get("sub") or result.get("user_id", "")
                self._token_cache[access_token] = {
                    "user_id": user_id,
                    "exp": time.time() + self._cache_ttl,
                }
                return user_id

            return None

        except (urllib.error.URLError, Exception) as e:
            import logging
            logging.getLogger(__name__).warning(f"SSO verification failed: {e}")
            return None

    def clear_cache(self):
        """清除 Token 缓存"""
        self._token_cache.clear()


class TokenAuthMiddleware:
    """SC-6: Token 验证中间件

    在 API 请求处理前进行身份验证:
    1. 从请求头 Authorization 中提取 Bearer Token
    2. 验证 Token 有效性
    3. 从 Token 中提取 user_id
    4. 替换请求参数中的 user_id
    5. 未认证请求返回标准错误
    """

    def __init__(self, auth_token: AuthToken, sso_service: Optional[SSOAuthService] = None):
        self._auth_token = auth_token
        self._sso_service = sso_service

    async def authenticate(self, request: Any) -> str:
        """验证请求并返回 user_id

        Raises:
            UnauthorizedError: 未认证或 Token 无效
            TokenExpiredError: Token 已过期
        """
        # 从请求头提取 Authorization
        auth_header = getattr(request, "authorization", "") or ""
        if not auth_header:
            raise UnauthorizedError("Missing Authorization header")

        # 解析 Bearer Token
        if not auth_header.startswith("Bearer "):
            raise UnauthorizedError("Invalid Authorization header format, expected Bearer token")

        token = auth_header[7:]

        try:
            # 优先使用本地 Token 验证
            return self._auth_token.extract_user_id(token)
        except TokenExpiredError:
            # Token 过期，尝试刷新
            raise
        except UnauthorizedError:
            # 本地验证失败，尝试 SSO 验证
            if self._sso_service:
                user_id = await self._sso_service.verify_sso_token(token)
                if user_id:
                    return user_id
            raise

    @staticmethod
    def get_standard_error_response(error: Exception) -> dict[str, Any]:
        """SC-6: 返回标准错误响应"""
        if isinstance(error, UnauthorizedError):
            return {
                "success": False,
                "error": "UNAUTHORIZED",
                "message": str(error),
                "error_code": 401,
                "timestamp": datetime.now().isoformat(),
            }
        elif isinstance(error, TokenExpiredError):
            return {
                "success": False,
                "error": "TOKEN_EXPIRED",
                "message": str(error),
                "error_code": 401,
                "timestamp": datetime.now().isoformat(),
                "hint": "Token has expired, please refresh or re-authenticate",
            }
        return {
            "success": False,
            "error": "UNKNOWN",
            "message": str(error),
            "error_code": 500,
            "timestamp": datetime.now().isoformat(),
        }


def require_auth(func):
    """SC-6: 认证装饰器 - 自动从 Token 提取 user_id

    用法:
        @require_auth
        async def get_profile(self, request) -> ApiResponse:
            ...

    自动从请求中提取 Token，验证后将 user_id 设置到 request 对象。
    """
    @functools.wraps(func)
    async def wrapper(self, request, *args, **kwargs):
        middleware = getattr(self, "_auth_middleware", None)
        if middleware is None:
            # 无认证中间件时跳过验证
            return await func(self, request, *args, **kwargs)

        try:
            user_id = await middleware.authenticate(request)
            # 将认证后的 user_id 设置到 request (如果 request 有 user_id 属性)
            if hasattr(request, "user_id"):
                request.user_id = user_id
            request._auth_user_id = user_id
            return await func(self, request, *args, **kwargs)
        except (UnauthorizedError, TokenExpiredError) as e:
            return ApiResponse(
                success=False,
                error=TokenAuthMiddleware.get_standard_error_response(e),
            )

    return wrapper


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
class FeedbackRequest:
    """用户反馈请求"""
    user_id: str
    prediction_id: str
    feedback_type: str = ""  # "like" 或 "dislike"
    prediction_data: Optional[dict[str, Any]] = None
    comment: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchFeedbackRequest:
    """批量反馈请求"""
    feedbacks: list[FeedbackRequest] = field(default_factory=list)


@dataclass
class MyDataQueryRequest:
    """个人数据查询请求"""
    user_id: str
    category: Optional[str] = None  # "behavior", "preference", "entity", None=全部
    time_range_days: Optional[int] = None
    page: int = 1
    page_size: int = 20


@dataclass
class DeleteDataRequest:
    """单条数据删除请求"""
    user_id: str
    data_id: str
    confirm: bool = False  # 二次确认


@dataclass
class BatchOperationRequest:
    """批量操作请求 (FE-6)"""
    user_id: str
    operations: list[dict[str, Any]] = field(default_factory=list)
    # 每个操作: {"action": "create|update|query", "data": {...}}
    max_batch_size: int = 100


@dataclass
class BatchOperationResponse:
    """批量操作响应"""
    success: bool
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExportDataRequest:
    """数据导出请求 (FE-7)"""
    user_id: str
    format: str = "json"       # "json" 或 "csv"
    categories: list[str] = field(default_factory=list)  # 空列表=全部
    time_range_days: Optional[int] = None
    include_metadata: bool = True


@dataclass
class TeamCreateRequest:
    """团队创建请求"""
    name: str
    description: str = ""
    department: str = ""
    created_by: str = ""
    members: list[dict[str, Any]] = field(default_factory=list)
    # 每个成员: {"user_id": "xxx", "role": "member", "permission": "view"}


@dataclass
class TeamMemoryCreateRequest:
    """团队记忆创建请求"""
    team_id: str
    title: str
    memory_type: str = "general"
    content: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    importance: int = 3
    is_public: bool = True
    allowed_members: list[str] = field(default_factory=list)


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

    SC-6: 支持 Token 认证中间件集成

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

    POST /api/v1/memory/feedback
        - 提交用户反馈（点赞/踩）
        - Request: FeedbackRequest
        - Response: {"feedback_id": "..."}

    POST /api/v1/memory/feedback/batch
        - 批量提交用户反馈
        - Request: BatchFeedbackRequest
        - Response: {"count": N}

    GET /api/v1/memory/feedback/stats
        - 获取反馈统计
        - Query: ?user_id=xxx&days=30
        - Response: {"total": N, "like_rate": 0.85, ...}

    POST /api/v1/memory/my-data
        - 查询个人数据清单
        - Request: MyDataQueryRequest
        - Response: {"categories": {...}, "items": [...]}

    DELETE /api/v1/memory/my-data
        - 删除单条个人数据
        - Request: DeleteDataRequest
        - Response: {"deleted": true}

    POST /api/v1/memory/batch
        - 批量操作（创建/更新/查询）
        - Request: BatchOperationRequest
        - Response: BatchOperationResponse

    POST /api/v1/memory/export
        - 导出个人记忆数据
        - Request: ExportDataRequest
        - Response: 结构化数据（JSON/CSV）或异步任务ID

    POST /api/v1/team
        - 创建团队
        - Request: TeamCreateRequest
        - Response: Team

    POST /api/v1/team/memory
        - 创建团队记忆
        - Request: TeamMemoryCreateRequest
        - Response: TeamMemory

    GET /api/v1/team/{team_id}/memories
        - 查询团队记忆
        - Query: ?user_id=xxx&memory_type=xxx
        - Response: list[TeamMemory]

    SC-6 新增:
    POST /api/v1/auth/token
        - 生成认证 Token
        - Request: {"user_id": "xxx"}
        - Response: {"token": "xxx", "expires_in": 3600}

    POST /api/v1/auth/refresh
        - 刷新 Token
        - Request: {"token": "xxx"}
        - Response: {"token": "xxx", "expires_in": 3600}"""

    def __init__(
        self,
        memory_manager,
        auth_token: Optional[AuthToken] = None,
        sso_service: Optional[SSOAuthService] = None,
    ):
        self._manager = memory_manager
        self._auth_token = auth_token
        self._auth_middleware: Optional[TokenAuthMiddleware] = None

        if auth_token:
            self._auth_middleware = TokenAuthMiddleware(auth_token, sso_service)

    # ─── SC-6: 认证相关接口 ─────────────────────────────

    async def generate_auth_token(self, user_id: str) -> ApiResponse:
        """SC-6: 生成认证 Token"""
        if not self._auth_token:
            return ApiResponse(success=False, error="Auth service not configured")
        try:
            token = self._auth_token.generate_token(user_id)
            return ApiResponse(success=True, data={
                "token": token,
                "expires_in": 3600,
                "token_type": "Bearer",
            })
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def refresh_auth_token(self, token: str) -> ApiResponse:
        """SC-6: 刷新认证 Token"""
        if not self._auth_token:
            return ApiResponse(success=False, error="Auth service not configured")
        try:
            if self._auth_token.should_refresh(token):
                new_token = self._auth_token.refresh_token(token)
                return ApiResponse(success=True, data={
                    "token": new_token,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                })
            return ApiResponse(success=False, error="Token does not need refresh")
        except (TokenExpiredError, UnauthorizedError) as e:
            return ApiResponse(success=False, error=str(e))
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

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

    async def submit_feedback(
        self, request: FeedbackRequest
    ) -> ApiResponse:
        """提交用户反馈（点赞/踩）

        反馈数据用于动态调整模式识别的置信度权重算法。
        接口响应时间 ≤ 300ms。
        """
        try:
            feedback_id = await self._manager.record_feedback(
                user_id=request.user_id,
                prediction_id=request.prediction_id,
                feedback_type=request.feedback_type,
                prediction_data=request.prediction_data,
                comment=request.comment,
            )
            return ApiResponse(
                success=True, data={"feedback_id": feedback_id}
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def submit_batch_feedback(
        self, request: BatchFeedbackRequest
    ) -> ApiResponse:
        """批量提交用户反馈"""
        try:
            count = 0
            for fb in request.feedbacks:
                await self._manager.record_feedback(
                    user_id=fb.user_id,
                    prediction_id=fb.prediction_id,
                    feedback_type=fb.feedback_type,
                    prediction_data=fb.prediction_data,
                    comment=fb.comment,
                )
                count += 1
            return ApiResponse(
                success=True, data={"count": count}
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_feedback_stats(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> ApiResponse:
        """获取反馈统计数据"""
        try:
            stats = await self._manager.get_feedback_stats(
                user_id=user_id, days=days
            )
            return ApiResponse(success=True, data=stats)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_my_data(
        self, request: MyDataQueryRequest
    ) -> ApiResponse:
        """查询个人数据清单

        按类别分组展示：行为记录/偏好设置/关联实体等。
        支持按时间范围筛选，实现数据权限校验。
        """
        try:
            result = await self._manager.get_user_data_inventory(
                user_id=request.user_id,
                category=request.category,
                time_range_days=request.time_range_days,
                page=request.page,
                page_size=request.page_size,
            )
            return ApiResponse(success=True, data=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def delete_my_data(
        self, request: DeleteDataRequest
    ) -> ApiResponse:
        """删除单条个人数据

        需要二次确认（confirm=True），删除后不可恢复。
        实现数据权限校验，确保用户仅能删除本人数据。
        """
        if not request.confirm:
            return ApiResponse(
                success=False,
                error="需要二次确认删除操作，请设置 confirm=True",
            )

        try:
            deleted = await self._manager.delete_user_data_item(
                user_id=request.user_id,
                data_id=request.data_id,
            )
            return ApiResponse(success=True, data={"deleted": deleted})
        except PermissionError as e:
            return ApiResponse(success=False, error=str(e))
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    # ─── FE-6: 批量操作 ──────────────────────────────────

    async def batch_operations(
        self, request: BatchOperationRequest
    ) -> BatchOperationResponse:
        """批量操作（创建/更新/查询）

        支持单次最多100条记录的批量操作。
        提供完整的错误处理与部分成功机制。
        响应时间比同等数量单条操作减少>60%。
        """
        if len(request.operations) > request.max_batch_size:
            return BatchOperationResponse(
                success=False,
                total=len(request.operations),
                errors=[{
                    "index": -1,
                    "error": f"批量操作数量超过限制({request.max_batch_size})",
                }],
            )

        results = []
        errors = []
        succeeded = 0
        failed = 0

        for i, op in enumerate(request.operations):
            action = op.get("action", "")
            data = op.get("data", {})

            try:
                if action == "create":
                    # 批量创建交互记录
                    from ..models.memory import InteractionRecord, SceneContext
                    scene = None
                    if data.get("scene_context"):
                        scene = SceneContext(**data["scene_context"])

                    interaction = InteractionRecord(
                        user_id=request.user_id,
                        interaction_type=data.get("interaction_type", ""),
                        intent=data.get("intent", ""),
                        raw_input=data.get("raw_input", ""),
                        processed_input=data.get("processed_input", {}),
                        scene_context=scene,
                        system_response=data.get("system_response", {}),
                        response_time_ms=data.get("response_time_ms", 0.0),
                        was_successful=data.get("was_successful", True),
                        session_id=data.get("session_id", ""),
                    )
                    memory_id = await self._manager.record_interaction(interaction)
                    results.append({"index": i, "action": "create", "memory_id": memory_id})
                    succeeded += 1

                elif action == "update":
                    # 批量更新记忆
                    memory_id = data.get("memory_id", "")
                    item = await self._manager.retrieve_memory(memory_id)
                    if item:
                        if "content" in data:
                            item.content.update(data["content"])
                        if "tags" in data:
                            item.tags = data["tags"]
                        await self._manager.store_memory(item)
                        results.append({"index": i, "action": "update", "memory_id": memory_id})
                        succeeded += 1
                    else:
                        errors.append({"index": i, "error": f"记忆不存在: {memory_id}"})
                        failed += 1

                elif action == "query":
                    # 批量查询
                    result = await self._manager.query_memories(
                        user_id=request.user_id,
                        keywords=data.get("keywords", []),
                        tags=data.get("tags", []),
                        max_results=data.get("max_results", 10),
                        memory_types=data.get("memory_types"),
                    )
                    results.append({
                        "index": i,
                        "action": "query",
                        "items": [item.to_dict() for item in result.items],
                        "total_found": result.total_found,
                    })
                    succeeded += 1

                else:
                    errors.append({"index": i, "error": f"未知操作类型: {action}"})
                    failed += 1

            except Exception as e:
                errors.append({"index": i, "error": str(e)})
                failed += 1

        return BatchOperationResponse(
            success=failed == 0,
            total=len(request.operations),
            succeeded=succeeded,
            failed=failed,
            results=results,
            errors=errors,
        )

    # ─── FE-7: 记忆导出 ──────────────────────────────────

    async def export_data(
        self, request: ExportDataRequest
    ) -> ApiResponse:
        """导出个人记忆数据

        支持JSON/CSV格式，符合GDPR数据可携带权要求。
        导出文件超过10MB时自动启用异步处理。
        """
        import json
        import csv
        import io
        import asyncio

        try:
            # 收集用户数据
            inventory = await self._manager.get_user_data_inventory(
                user_id=request.user_id,
                category=None,
                time_range_days=request.time_range_days,
                page=1,
                page_size=10000,  # 获取全部数据
            )

            # 按类别筛选
            if request.categories:
                inventory["items"] = [
                    item for item in inventory["items"]
                    if item["category"] in request.categories
                ]

            export_data = {
                "user_id": request.user_id,
                "exported_at": datetime.now().isoformat(),
                "format": request.format,
                "total_items": len(inventory["items"]),
                "categories": inventory.get("categories", {}),
                "items": inventory["items"],
            }

            if not request.include_metadata:
                export_data.pop("categories", None)

            # 估算数据大小
            data_json = json.dumps(export_data, ensure_ascii=False, default=str)
            data_size_mb = len(data_json.encode("utf-8")) / (1024 * 1024)

            # 大文件异步处理
            if data_size_mb > 10:
                task_id = f"export_{request.user_id}_{int(datetime.now().timestamp())}"
                asyncio.create_task(self._async_export(task_id, export_data, request.format))
                return ApiResponse(
                    success=True,
                    data={
                        "async": True,
                        "task_id": task_id,
                        "estimated_size_mb": round(data_size_mb, 2),
                        "message": "数据量较大，已启用异步处理，完成后将通知您",
                    },
                )

            if request.format == "csv":
                csv_content = self._convert_to_csv(inventory["items"])
                return ApiResponse(
                    success=True,
                    data={
                        "format": "csv",
                        "content": csv_content,
                        "total_items": len(inventory["items"]),
                    },
                )
            else:
                return ApiResponse(
                    success=True,
                    data=export_data,
                )

        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def _async_export(
        self, task_id: str, data: dict[str, Any], format: str
    ):
        """异步导出处理（后台任务）"""
        import json
        import asyncio

        await asyncio.sleep(1)  # 模拟处理时间
        # 实际场景中，此处将数据写入文件并通知用户
        logger = __import__("logging").getLogger(__name__)
        logger.info(f"Async export completed: {task_id}, items: {len(data.get('items', []))}")

    def _convert_to_csv(self, items: list[dict[str, Any]]) -> str:
        """将数据转换为CSV格式"""
        import csv
        import io

        if not items:
            return ""

        output = io.StringIO()
        # 收集所有可能的字段
        all_keys = set()
        for item in items:
            all_keys.update(item.keys())
        fieldnames = sorted(all_keys)

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            # 将复杂类型转为字符串
            row = {}
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    import json
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v
            writer.writerow(row)

        return output.getvalue()

    # ─── FE-1: 团队协作记忆 ──────────────────────────────

    async def create_team(
        self, request: TeamCreateRequest
    ) -> ApiResponse:
        """创建团队"""
        from ..models.team_memory import Team, TeamMember, TeamPermission
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)

            members = []
            for m in request.members:
                members.append(TeamMember(
                    user_id=m["user_id"],
                    role=m.get("role", "member"),
                    permission=TeamPermission(m.get("permission", "view")),
                ))

            team = Team(
                name=request.name,
                description=request.description,
                department=request.department,
                members=members,
                created_by=request.created_by,
            )

            await team_store.create_team(team)
            return ApiResponse(success=True, data=team.to_dict())
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def get_user_teams(self, user_id: str) -> ApiResponse:
        """获取用户所属团队列表"""
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)
            teams = await team_store.get_user_teams(user_id)
            return ApiResponse(
                success=True,
                data=[t.to_dict() for t in teams],
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def create_team_memory(
        self, request: TeamMemoryCreateRequest
    ) -> ApiResponse:
        """创建团队记忆"""
        from ..models.team_memory import TeamMemory, TeamMemoryType
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)

            # 权限校验：检查用户是否为团队成员且有编辑权限
            team = await team_store.get_team(request.team_id)
            if team is None:
                return ApiResponse(success=False, error="团队不存在")
            if not team.can_edit(request.created_by):
                return ApiResponse(success=False, error="无编辑权限")

            memory = TeamMemory(
                team_id=request.team_id,
                title=request.title,
                memory_type=TeamMemoryType(request.memory_type),
                content=request.content,
                created_by=request.created_by,
                tags=request.tags,
                keywords=request.keywords,
                importance=request.importance,
                is_public=request.is_public,
                allowed_members=request.allowed_members,
            )

            await team_store.store_memory(memory)
            return ApiResponse(success=True, data=memory.to_dict())
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def query_team_memories(
        self,
        team_id: str,
        user_id: str = "",
        memory_type: str = "",
        max_results: int = 20,
    ) -> ApiResponse:
        """查询团队记忆"""
        from ..models.team_memory import TeamMemoryQuery, TeamMemoryType
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)

            memory_types = None
            if memory_type:
                memory_types = [TeamMemoryType(memory_type)]

            query = TeamMemoryQuery(
                team_id=team_id,
                user_id=user_id,
                memory_types=memory_types,
                max_results=max_results,
            )

            result = await team_store.query_memories(query)
            return ApiResponse(success=True, data=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def update_team_memory(
        self, memory_id: str, data: dict[str, Any], user_id: str
    ) -> ApiResponse:
        """更新团队记忆"""
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)
            memory = await team_store.get_memory(memory_id)
            if memory is None:
                return ApiResponse(success=False, error="记忆不存在")

            # 权限校验
            team = await team_store.get_team(memory.team_id)
            if team is None or not team.can_edit(user_id):
                return ApiResponse(success=False, error="无编辑权限")

            if "title" in data:
                memory.title = data["title"]
            if "content" in data:
                memory.content.update(data["content"])
            if "tags" in data:
                memory.tags = data["tags"]
            if "keywords" in data:
                memory.keywords = data["keywords"]
            memory.updated_by = user_id

            await team_store.update_memory(memory)
            return ApiResponse(success=True, data=memory.to_dict())
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    async def delete_team_memory(
        self, memory_id: str, user_id: str
    ) -> ApiResponse:
        """删除团队记忆"""
        from ..storage.team_store import TeamStore

        try:
            team_store = TeamStore(self._manager._long_term._db_path)
            memory = await team_store.get_memory(memory_id)
            if memory is None:
                return ApiResponse(success=False, error="记忆不存在")

            # 权限校验
            team = await team_store.get_team(memory.team_id)
            if team is None or not team.can_edit(user_id):
                return ApiResponse(success=False, error="无编辑权限")

            deleted = await team_store.delete_memory(memory_id)
            return ApiResponse(success=True, data={"deleted": deleted})
        except Exception as e:
            return ApiResponse(success=False, error=str(e))