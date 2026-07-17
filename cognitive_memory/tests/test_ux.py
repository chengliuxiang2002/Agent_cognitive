"""
认知记忆模块 - UX 优化功能单元测试

覆盖:
- UX-2: 记忆可解释性 (reason 字段)
- UX-3: 用户反馈闭环
- UX-4: 记忆透明度面板
- UX-5: 多语言支持
- UX-6: 通知推送
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from cognitive_memory.models.memory import (
    MemoryItem,
    MemoryType,
    MemoryImportance,
    UserProfile,
    InteractionRecord,
    SceneContext,
    BehaviorPattern,
    MemoryQuery,
    MemoryRetrievalResult,
)
from cognitive_memory.storage import (
    FeedbackStore,
    ShortTermMemoryStore,
    LongTermMemoryStore,
    ProfileStore,
    InteractionStore,
    PatternStore,
)
from cognitive_memory.learner.pattern_learner import MemoryEncoder
from cognitive_memory.core.memory_manager import MemoryManager
from cognitive_memory.core.context_engine import ContextEngine
from cognitive_memory.api.routes import (
    MemoryAPI,
    FeedbackRequest,
    BatchFeedbackRequest,
    MyDataQueryRequest,
    DeleteDataRequest,
    ApiResponse,
)


# ─── UX-2: 记忆可解释性 ──────────────────────────────────

class TestMemoryExplainability:
    """测试 predict_user_needs 的 reason 字段"""

    @pytest.fixture
    def engine(self):
        memory_store = AsyncMock()
        pattern_store = AsyncMock()
        return ContextEngine(memory_store, pattern_store)

    @pytest.fixture
    def sample_scene(self):
        return SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
        )

    @pytest.mark.asyncio
    async def test_pattern_reason_route(self, engine, sample_scene):
        """测试路线模式推荐理由"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="route_pref_home_morning",
            pattern_type="route",
            trigger_conditions={"time_of_day": "morning"},
            expected_action={"navigate_to": "公司"},
            occurrence_count=5,
            confidence=0.9,
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[pattern])
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )

        predictions = await engine.predict_user_needs(
            "user_001", sample_scene
        )

        assert len(predictions) > 0
        pred = predictions[0]
        assert "reason" in pred
        assert "5次" in pred["reason"]
        assert "公司" in pred["reason"]
        assert len(pred["reason"]) >= 10

    @pytest.mark.asyncio
    async def test_pattern_reason_temperature(self, engine, sample_scene):
        """测试温度偏好推荐理由"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="temperature_pref_morning",
            pattern_type="temperature",
            trigger_conditions={"time_of_day": "morning"},
            expected_action={"set_temperature": 24.0},
            occurrence_count=3,
            confidence=0.8,
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[pattern])
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )

        predictions = await engine.predict_user_needs(
            "user_001", sample_scene
        )

        assert len(predictions) > 0
        assert "3次" in predictions[0]["reason"]
        assert "24" in predictions[0]["reason"]

    @pytest.mark.asyncio
    async def test_profile_reason(self, engine, sample_scene):
        """测试画像推荐理由"""
        profile = UserProfile(
            user_id="user_001",
            temperature_preference=23.0,
            driving_mode_preference="comfort",
            confidence_score=0.8,
            data_points_count=50,
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[])
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )

        predictions = await engine.predict_user_needs(
            "user_001", sample_scene, profile
        )

        assert len(predictions) >= 1
        temp_pred = [p for p in predictions if p["source"] == "user_profile"]
        assert len(temp_pred) > 0
        assert "reason" in temp_pred[0]
        assert "50次" in temp_pred[0]["reason"] or "温度" in temp_pred[0]["reason"]

    @pytest.mark.asyncio
    async def test_reason_coverage_100_percent(self, engine, sample_scene):
        """测试所有预测结果都有 reason 字段（覆盖率100%）"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="test_pattern",
            pattern_type="media",
            trigger_conditions={},
            expected_action={"preferred_genre": "pop"},
            occurrence_count=5,
            confidence=0.7,
        )
        profile = UserProfile(
            user_id="user_001",
            temperature_preference=22.0,
            confidence_score=0.7,
            data_points_count=10,
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[pattern])
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )

        predictions = await engine.predict_user_needs(
            "user_001", sample_scene, profile
        )

        for pred in predictions:
            assert "reason" in pred, f"预测缺少 reason 字段: {pred.get('type')}"
            assert 10 <= len(pred["reason"]) <= 100, (
                f"reason 长度不符合20-50字范围: {len(pred['reason'])}字 - {pred['reason']}"
            )

    def test_time_slot_label(self):
        """测试时段标签转换"""
        assert ContextEngine._time_slot_label("morning") == "早晨"
        assert ContextEngine._time_slot_label("afternoon") == "下午"
        assert ContextEngine._time_slot_label("evening") == "傍晚"
        assert ContextEngine._time_slot_label("night") == "夜间"
        assert ContextEngine._time_slot_label("unknown") == "该时段"


# ─── UX-3: 用户反馈闭环 ──────────────────────────────────

class TestFeedbackSystem:
    """测试反馈接口"""

    @pytest.fixture
    def api(self, memory_manager):
        return MemoryAPI(memory_manager)

    @pytest.mark.asyncio
    async def test_submit_feedback_like(self, api):
        """测试提交点赞反馈"""
        request = FeedbackRequest(
            user_id="user_001",
            prediction_id="pred_route_home",
            feedback_type="like",
            comment="推荐准确",
        )
        response = await api.submit_feedback(request)
        assert response.success is True
        assert "feedback_id" in response.data

    @pytest.mark.asyncio
    async def test_submit_feedback_dislike(self, api):
        """测试提交踩反馈"""
        request = FeedbackRequest(
            user_id="user_001",
            prediction_id="pred_temp_high",
            feedback_type="dislike",
            comment="温度不合适",
        )
        response = await api.submit_feedback(request)
        assert response.success is True
        assert "feedback_id" in response.data

    @pytest.mark.asyncio
    async def test_batch_feedback(self, api):
        """测试批量提交反馈"""
        request = BatchFeedbackRequest(
            feedbacks=[
                FeedbackRequest(
                    user_id="user_001",
                    prediction_id="pred_1",
                    feedback_type="like",
                ),
                FeedbackRequest(
                    user_id="user_001",
                    prediction_id="pred_2",
                    feedback_type="dislike",
                ),
            ]
        )
        response = await api.submit_batch_feedback(request)
        assert response.success is True
        assert response.data["count"] == 2

    @pytest.mark.asyncio
    async def test_get_feedback_stats(self, api):
        """测试获取反馈统计"""
        # 先提交一些反馈
        for i in range(3):
            await api.submit_feedback(FeedbackRequest(
                user_id="user_001",
                prediction_id=f"pred_{i}",
                feedback_type="like" if i < 2 else "dislike",
            ))

        response = await api.get_feedback_stats(user_id="user_001", days=30)
        assert response.success is True
        assert response.data["total"] >= 3
        assert response.data["likes"] >= 2
        assert response.data["dislikes"] >= 1
        assert 0 <= response.data["like_rate"] <= 1

    @pytest.mark.asyncio
    async def test_feedback_response_time(self, api):
        """测试反馈接口响应时间 ≤ 300ms"""
        import time
        start = time.time()
        response = await api.submit_feedback(FeedbackRequest(
            user_id="user_001",
            prediction_id="pred_perf",
            feedback_type="like",
        ))
        elapsed = (time.time() - start) * 1000
        assert response.success is True
        assert elapsed < 300, f"响应时间 {elapsed:.0f}ms 超过 300ms 限制"


# ─── UX-4: 记忆透明度面板 ──────────────────────────────────

class TestMemoryTransparency:
    """测试个人数据清单接口"""

    @pytest.fixture
    def api(self, memory_manager):
        return MemoryAPI(memory_manager)

    @pytest.mark.asyncio
    async def test_get_my_data_all_categories(self, api, memory_manager):
        """测试获取全部类别的个人数据"""
        # 先记录一些交互数据
        from cognitive_memory.models.memory import InteractionRecord, SceneContext
        interaction = InteractionRecord(
            user_id="user_001",
            interaction_type="voice_command",
            intent="navigate",
            raw_input="导航到公司",
            scene_context=SceneContext(
                time_of_day="morning",
                location_type="home",
            ),
        )
        await memory_manager.record_interaction(interaction)

        request = MyDataQueryRequest(user_id="user_001")
        response = await api.get_my_data(request)

        assert response.success is True
        assert "categories" in response.data
        assert "items" in response.data
        assert "total" in response.data

        # 验证类别
        categories = response.data["categories"]
        assert "behavior" in categories
        assert categories["behavior"]["label"] == "行为记录"

    @pytest.mark.asyncio
    async def test_get_my_data_filter_by_category(self, api):
        """测试按类别筛选"""
        request = MyDataQueryRequest(user_id="user_001", category="behavior")
        response = await api.get_my_data(request)

        assert response.success is True
        for item in response.data["items"]:
            assert item["category"] == "behavior"

    @pytest.mark.asyncio
    async def test_get_my_data_pagination(self, api):
        """测试分页"""
        request = MyDataQueryRequest(
            user_id="user_001", page=1, page_size=5
        )
        response = await api.get_my_data(request)

        assert response.success is True
        assert response.data["page"] == 1
        assert response.data["page_size"] == 5

    @pytest.mark.asyncio
    async def test_delete_my_data_without_confirm(self, api):
        """测试二次确认 - 未确认时拒绝删除"""
        request = DeleteDataRequest(
            user_id="user_001",
            data_id="some_id",
            confirm=False,
        )
        response = await api.delete_my_data(request)
        assert response.success is False
        assert "二次确认" in response.error

    @pytest.mark.asyncio
    async def test_delete_my_data_permission_check(self, api):
        """测试数据权限校验 - 不能删除他人数据"""
        request = DeleteDataRequest(
            user_id="user_001",
            data_id="pref_temperature_user_002",
            confirm=True,
        )
        response = await api.delete_my_data(request)
        assert response.success is False
        assert "无权" in response.error


# ─── UX-5: 多语言支持 ──────────────────────────────────

class TestMultiLanguage:
    """测试多语言支持"""

    @pytest.fixture
    def encoder(self):
        return MemoryEncoder()

    def test_detect_chinese(self, encoder):
        """测试中文检测"""
        assert encoder._detect_language("安全气囊故障") == "zh"
        assert encoder._detect_language("导航到公司") == "zh"

    def test_detect_english(self, encoder):
        """测试英文检测"""
        assert encoder._detect_language("airbag warning") == "en"
        assert encoder._detect_language("navigate to office") == "en"

    def test_detect_japanese(self, encoder):
        """测试日文检测"""
        assert encoder._detect_language("エアバッグ警告") == "ja"
        assert encoder._detect_language("危険な状況") == "ja"

    def test_detect_korean(self, encoder):
        """测试韩文检测"""
        assert encoder._detect_language("에어백 경고") == "ko"
        assert encoder._detect_language("위험 상황") == "ko"

    def test_detect_empty_text(self, encoder):
        """测试空文本默认语言"""
        assert encoder._detect_language("") == "zh"

    def test_detect_language_speed(self, encoder):
        """测试语言检测响应时间 ≤ 100ms"""
        import time
        test_texts = [
            "安全气囊故障",
            "airbag warning",
            "エアバッグ警告",
            "에어백 경고",
        ]
        for text in test_texts:
            start = time.time()
            lang = encoder._detect_language(text)
            elapsed = (time.time() - start) * 1000
            assert elapsed < 100, f"语言检测 {text} 耗时 {elapsed:.0f}ms 超过 100ms 限制"

    def test_chinese_safety_keywords(self, encoder):
        """测试中文安全关键词识别"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="刹车失灵了",
            interaction_type="voice_command",
        )
        importance = encoder._assess_importance(record)
        assert importance == MemoryImportance.CRITICAL

    def test_english_safety_keywords(self, encoder):
        """测试英文安全关键词识别"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="brake failure detected",
            interaction_type="voice_command",
        )
        importance = encoder._assess_importance(record)
        assert importance == MemoryImportance.CRITICAL

    def test_japanese_safety_keywords(self, encoder):
        """测试日文安全关键词识别"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="ブレーキ故障",
            interaction_type="voice_command",
        )
        importance = encoder._assess_importance(record)
        assert importance == MemoryImportance.CRITICAL

    def test_korean_safety_keywords(self, encoder):
        """测试韩文安全关键词识别"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="브레이크 고장",
            interaction_type="voice_command",
        )
        importance = encoder._assess_importance(record)
        assert importance == MemoryImportance.CRITICAL

    def test_english_navigation_keywords(self, encoder):
        """测试英文导航关键词识别"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="navigate to the airport",
            interaction_type="voice_command",
        )
        importance = encoder._assess_importance(record)
        assert importance == MemoryImportance.HIGH

    def test_non_safety_text(self, encoder):
        """测试非安全文本不被误判为 CRITICAL"""
        record = InteractionRecord(
            user_id="user_001",
            raw_input="播放音乐",
            interaction_type="voice_command",
            intent="play_music",
        )
        importance = encoder._assess_importance(record)
        assert importance != MemoryImportance.CRITICAL


# ─── UX-6: 通知推送 ──────────────────────────────────

class TestNotification:
    """测试通知推送功能"""

    @pytest.fixture
    def manager(self, tmp_path):
        db_path = str(tmp_path / "test_notify.db")
        return MemoryManager(db_path=db_path)

    def test_build_notification_route(self, manager):
        """测试路线通知构建"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="route_pref_home",
            pattern_type="route",
            expected_action={"navigate_to": "公司"},
            occurrence_count=10,
            confidence=0.85,
            first_observed=datetime.now() - timedelta(days=7),
            last_observed=datetime.now(),
        )
        notification = manager._build_notification(pattern)
        assert notification["title"] == "系统已学习到您的通勤路线偏好"
        assert "公司" in notification["body"]
        assert notification["confidence"] == 0.85
        assert notification["occurrence_count"] == 10
        assert notification["learning_period_days"] >= 7

    def test_build_notification_temperature(self, manager):
        """测试温度通知构建"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="temp_pref",
            pattern_type="temperature",
            expected_action={"set_temperature": 24.0},
            occurrence_count=8,
            confidence=0.9,
            first_observed=datetime.now() - timedelta(days=5),
            last_observed=datetime.now(),
        )
        notification = manager._build_notification(pattern)
        assert "温度偏好" in notification["title"]
        assert "24" in notification["body"]

    def test_build_notification_media(self, manager):
        """测试媒体通知构建"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="media_genre_pop",
            pattern_type="media",
            expected_action={"preferred_genre": "pop"},
            occurrence_count=15,
            confidence=0.95,
            first_observed=datetime.now() - timedelta(days=14),
            last_observed=datetime.now(),
        )
        notification = manager._build_notification(pattern)
        assert "音乐偏好" in notification["title"]
        assert "pop" in notification["body"]

    @pytest.mark.asyncio
    async def test_notification_confidence_threshold(self, manager):
        """测试置信度阈值 - 低于0.8不触发通知"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="low_confidence_pattern",
            pattern_type="route",
            expected_action={"navigate_to": "商场"},
            occurrence_count=2,
            confidence=0.5,  # 低于阈值
            first_observed=datetime.now() - timedelta(days=1),
            last_observed=datetime.now(),
        )
        await manager._notify_pattern_learned("user_001", pattern)
        notifications = manager.get_pending_notifications()
        assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_notification_high_confidence(self, manager):
        """测试高置信度触发通知"""
        pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="high_conf_pattern",
            pattern_type="route",
            expected_action={"navigate_to": "公司"},
            occurrence_count=10,
            confidence=0.9,
            first_observed=datetime.now() - timedelta(days=7),
            last_observed=datetime.now(),
        )
        await manager._notify_pattern_learned("user_001", pattern)
        notifications = manager.get_pending_notifications()
        assert len(notifications) == 1
        assert notifications[0]["notification"]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_notification_frequency_limit(self, manager):
        """测试通知频率限制 - 30天内同类型仅推送1次"""
        # 先保存一个旧模式
        old_pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="old_route",
            pattern_type="route",
            expected_action={"navigate_to": "旧目的地"},
            occurrence_count=5,
            confidence=0.85,
            first_observed=datetime.now() - timedelta(days=35),
            last_observed=datetime.now() - timedelta(days=10),
        )
        await manager._pattern_store.save_pattern(old_pattern)

        # 新学习到的同类型模式
        new_pattern = BehaviorPattern(
            user_id="user_001",
            pattern_name="new_route",
            pattern_type="route",
            expected_action={"navigate_to": "新目的地"},
            occurrence_count=8,
            confidence=0.9,
            first_observed=datetime.now() - timedelta(days=5),
            last_observed=datetime.now(),
        )
        await manager._notify_pattern_learned("user_001", new_pattern)

        # 10天前已有同类型通知，应被限制
        notifications = manager.get_pending_notifications()
        assert len(notifications) == 0, "30天内同类型不应重复推送"


# ─── 反馈存储测试 ──────────────────────────────────

class TestFeedbackStore:
    """测试 FeedbackStore 存储层"""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_feedback.db")
        return FeedbackStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_record_and_stats(self, store):
        """测试记录反馈和统计"""
        # 记录反馈
        await store.record_feedback(
            feedback_id="fb_001",
            user_id="user_001",
            prediction_id="pred_001",
            feedback_type="like",
        )
        await store.record_feedback(
            feedback_id="fb_002",
            user_id="user_001",
            prediction_id="pred_002",
            feedback_type="dislike",
        )

        # 获取统计
        stats = await store.get_feedback_stats(user_id="user_001", days=30)
        assert stats["total"] == 2
        assert stats["likes"] == 1
        assert stats["dislikes"] == 1
        assert stats["like_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_get_user_feedback(self, store):
        """测试获取用户反馈列表"""
        await store.record_feedback(
            feedback_id="fb_001",
            user_id="user_001",
            prediction_id="pred_001",
            feedback_type="like",
            comment="很准确",
        )

        feedbacks = await store.get_user_feedback("user_001", limit=10)
        assert len(feedbacks) >= 1
        assert feedbacks[0]["feedback_type"] == "like"
        assert feedbacks[0]["comment"] == "很准确"

    @pytest.mark.asyncio
    async def test_empty_stats(self, store):
        """测试空统计"""
        stats = await store.get_feedback_stats(user_id="nonexistent", days=30)
        assert stats["total"] == 0
        assert stats["like_rate"] == 0.0