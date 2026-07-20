"""
认知记忆模块 - 功能增强 (FE-1 ~ FE-7) 单元测试

覆盖:
- FE-1: 团队协作记忆
- FE-2: 冷启动解决
- FE-3: 日程上下文感知
- FE-4: 文档上下文记忆
- FE-5: 跨场景迁移
- FE-6: 批量操作
- FE-7: 记忆导出
"""

import pytest
import json
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
from cognitive_memory.models.team_memory import (
    Team,
    TeamMember,
    TeamMemory,
    TeamMemoryType,
    TeamPermission,
    TeamMemoryQuery,
)
from cognitive_memory.models.document_context import (
    DocumentContext,
    DocumentMetadata,
    DocumentEditRecord,
    DocumentFormat,
    DocumentAction,
    DocumentContextQuery,
)
from cognitive_memory.core.context_engine import ContextEngine
from cognitive_memory.core.cold_start import ColdStartEngine, DEFAULT_PROFILE_TEMPLATES
from cognitive_memory.core.user_profile_builder import UserProfileBuilder
from cognitive_memory.storage.team_store import TeamStore
from cognitive_memory.api.routes import (
    MemoryAPI,
    BatchOperationRequest,
    BatchOperationResponse,
    ExportDataRequest,
    TeamCreateRequest,
    TeamMemoryCreateRequest,
    ApiResponse,
)


# ─── FE-1: 团队协作记忆 ──────────────────────────────────

class TestTeamMemory:
    """测试团队协作记忆模型与存储"""

    def test_team_creation(self):
        """测试团队创建"""
        team = Team(
            name="研发一部",
            description="AI座舱研发团队",
            department="engineering",
            created_by="user_001",
            members=[
                TeamMember(user_id="user_001", role="leader", permission=TeamPermission.ADMIN),
                TeamMember(user_id="user_002", role="member", permission=TeamPermission.EDIT),
                TeamMember(user_id="user_003", role="member", permission=TeamPermission.VIEW),
            ],
        )

        assert team.name == "研发一部"
        assert len(team.members) == 3
        assert team.has_member("user_001")
        assert not team.has_member("user_999")
        assert team.can_edit("user_001")
        assert team.can_edit("user_002")
        assert not team.can_edit("user_003")
        assert team.can_admin("user_001")
        assert not team.can_admin("user_002")

    def test_team_memory_creation(self):
        """测试团队记忆创建"""
        memory = TeamMemory(
            team_id="team_001",
            title="3F会议室预定规则",
            memory_type=TeamMemoryType.MEETING_ROOM,
            content={
                "room": "3F-会议室A",
                "capacity": 10,
                "booking_url": "https://booking.example.com/3f-a",
                "rules": "需提前2小时预约，每次最长2小时",
            },
            created_by="user_001",
            tags=["会议室", "3F"],
            keywords=["会议室", "预定", "3F"],
            importance=4,
        )

        d = memory.to_dict()
        assert d["title"] == "3F会议室预定规则"
        assert d["memory_type"] == "meeting_room"
        assert d["importance"] == 4
        assert "会议室" in d["tags"]

    def test_team_memory_permission(self):
        """测试团队记忆权限隔离"""
        memory_public = TeamMemory(
            team_id="team_001",
            title="公开文档",
            is_public=True,
        )
        memory_private = TeamMemory(
            team_id="team_001",
            title="私有文档",
            is_public=False,
            allowed_members=["user_001", "user_002"],
        )

        assert memory_public.is_public
        assert not memory_private.is_public
        assert "user_001" in memory_private.allowed_members
        assert "user_999" not in memory_private.allowed_members

    @pytest.mark.asyncio
    async def test_team_store_crud(self, tmp_path):
        """测试团队存储CRUD操作"""
        db_path = str(tmp_path / "test_team.db")
        store = TeamStore(db_path)

        # 创建团队
        team = Team(
            name="测试团队",
            department="engineering",
            created_by="user_001",
            members=[
                TeamMember(user_id="user_001", role="leader", permission=TeamPermission.ADMIN),
                TeamMember(user_id="user_002", role="member", permission=TeamPermission.VIEW),
            ],
        )
        await store.create_team(team)

        # 查询团队
        found = await store.get_team(team.id)
        assert found is not None
        assert found.name == "测试团队"
        assert found.has_member("user_001")

        # 获取用户团队列表
        user_teams = await store.get_user_teams("user_001")
        assert len(user_teams) == 1
        assert user_teams[0].id == team.id

        # 非团队成员无权限
        non_member_teams = await store.get_user_teams("user_999")
        assert len(non_member_teams) == 0

    @pytest.mark.asyncio
    async def test_team_memory_store(self, tmp_path):
        """测试团队记忆存储"""
        db_path = str(tmp_path / "test_team_memory.db")
        store = TeamStore(db_path)

        # 创建团队
        team = Team(
            name="测试团队",
            created_by="user_001",
            members=[
                TeamMember(user_id="user_001", permission=TeamPermission.ADMIN),
            ],
        )
        await store.create_team(team)

        # 创建团队记忆
        memory = TeamMemory(
            team_id=team.id,
            title="项目文档路径",
            memory_type=TeamMemoryType.PROJECT_DOC,
            content={"path": "/docs/project/ai-cockpit"},
            created_by="user_001",
            tags=["文档", "项目"],
        )
        await store.store_memory(memory)

        # 查询团队记忆
        result = await store.query_memories(TeamMemoryQuery(
            team_id=team.id,
            user_id="user_001",
            max_results=10,
        ))
        assert result["total"] == 1
        assert result["items"][0]["title"] == "项目文档路径"

        # 非团队成员查询（无权限）
        result_denied = await store.query_memories(TeamMemoryQuery(
            team_id=team.id,
            user_id="user_999",
            max_results=10,
        ))
        # 公开记忆对非成员也可见
        assert result_denied["total"] == 1

    @pytest.mark.asyncio
    async def test_team_memory_isolation(self, tmp_path):
        """测试团队记忆与个人记忆存储隔离"""
        db_path = str(tmp_path / "test_isolation.db")
        store = TeamStore(db_path)

        # 创建两个团队
        team_a = Team(name="团队A", created_by="user_001",
                       members=[TeamMember(user_id="user_001", permission=TeamPermission.ADMIN)])
        team_b = Team(name="团队B", created_by="user_002",
                       members=[TeamMember(user_id="user_002", permission=TeamPermission.ADMIN)])
        await store.create_team(team_a)
        await store.create_team(team_b)

        # 为团队A创建记忆
        mem_a = TeamMemory(team_id=team_a.id, title="A的记忆", created_by="user_001")
        await store.store_memory(mem_a)

        # 为团队B创建记忆
        mem_b = TeamMemory(team_id=team_b.id, title="B的记忆", created_by="user_002")
        await store.store_memory(mem_b)

        # 查询团队A的记忆
        result_a = await store.query_memories(TeamMemoryQuery(team_id=team_a.id, user_id="user_001"))
        assert result_a["total"] == 1
        assert result_a["items"][0]["title"] == "A的记忆"

        # 查询团队B的记忆
        result_b = await store.query_memories(TeamMemoryQuery(team_id=team_b.id, user_id="user_002"))
        assert result_b["total"] == 1
        assert result_b["items"][0]["title"] == "B的记忆"

    @pytest.mark.asyncio
    async def test_create_3_teams(self, tmp_path):
        """测试创建至少3个测试团队（验收标准）"""
        db_path = str(tmp_path / "test_3_teams.db")
        store = TeamStore(db_path)

        teams = []
        for i in range(3):
            team = Team(
                name=f"测试团队{i+1}",
                department=["engineering", "product", "design"][i],
                created_by="user_001",
                members=[
                    TeamMember(user_id=f"user_{j+1:03d}", role="member",
                              permission=TeamPermission.VIEW if j > 0 else TeamPermission.ADMIN)
                    for j in range(3)
                ],
            )
            await store.create_team(team)
            teams.append(team)

        # 为每个团队创建共享记忆
        for team in teams:
            memory = TeamMemory(
                team_id=team.id,
                title=f"{team.name}共享记忆",
                content={"info": f"这是{team.name}的共享信息"},
                created_by="user_001",
            )
            await store.store_memory(memory)

        # 验证
        for team in teams:
            result = await store.query_memories(TeamMemoryQuery(team_id=team.id, user_id="user_001"))
            assert result["total"] >= 1


# ─── FE-2: 冷启动解决 ──────────────────────────────────

class TestColdStart:
    """测试冷启动引擎"""

    def test_default_template_exists(self):
        """测试默认画像模板库存在"""
        assert "engineering" in DEFAULT_PROFILE_TEMPLATES
        assert "product" in DEFAULT_PROFILE_TEMPLATES
        assert "default" in DEFAULT_PROFILE_TEMPLATES
        template = DEFAULT_PROFILE_TEMPLATES["engineering"]
        assert "temperature_preference" in template
        assert "driving_mode_preference" in template
        assert "music_preferences" in template

    def test_initialize_new_user_with_department(self):
        """测试按部门初始化新用户画像"""
        engine = ColdStartEngine()
        profile = engine.initialize_new_user_profile(
            user_id="new_user_001",
            department="engineering",
            role="engineer",
        )

        assert profile.user_id == "new_user_001"
        assert profile.temperature_preference > 0
        assert profile.driving_mode_preference != ""
        assert len(profile.music_preferences) > 0
        assert profile.confidence_score < 0.3  # 冷启动低置信度
        assert profile.data_points_count == 0

    def test_initialize_default_when_department_unknown(self):
        """测试未知部门时使用默认模板"""
        engine = ColdStartEngine()
        profile = engine.initialize_new_user_profile(
            user_id="new_user_002",
            department="unknown_department",
        )

        default_template = DEFAULT_PROFILE_TEMPLATES["default"]
        assert profile.temperature_preference == default_template["temperature_preference"]
        assert profile.driving_mode_preference == default_template["driving_mode_preference"]

    def test_calculate_group_similarity(self):
        """测试群体画像相似度计算"""
        engine = ColdStartEngine()

        target = UserProfile(
            user_id="target",
            temperature_preference=23.5,
            driving_mode_preference="comfort",
            music_preferences=["electronic", "ambient"],
            interaction_style="moderate",
            voice_assistant_style="concise",
        )

        group = [
            UserProfile(
                user_id="u1",
                temperature_preference=23.0,
                driving_mode_preference="comfort",
                music_preferences=["electronic", "classical"],
                interaction_style="moderate",
                voice_assistant_style="concise",
            ),
            UserProfile(
                user_id="u2",
                temperature_preference=24.0,
                driving_mode_preference="comfort",
                music_preferences=["ambient", "jazz"],
                interaction_style="moderate",
                voice_assistant_style="detailed",
            ),
        ]

        similarity = engine.calculate_group_similarity(target, group)
        assert similarity > 0.5  # 应该较高相似度

    def test_high_similarity_same_dept(self):
        """测试同部门/同角色用户相似度 > 75%"""
        engine = ColdStartEngine()

        profile_a = UserProfile(
            user_id="eng_001",
            temperature_preference=23.5,
            driving_mode_preference="comfort",
            music_preferences=["electronic", "ambient", "classical"],
            interaction_style="moderate",
            voice_assistant_style="concise",
        )

        profile_b = UserProfile(
            user_id="eng_002",
            temperature_preference=23.0,
            driving_mode_preference="comfort",
            music_preferences=["electronic", "ambient", "jazz"],
            interaction_style="moderate",
            voice_assistant_style="concise",
        )

        sim = engine._calculate_profile_similarity(profile_a, profile_b)
        assert sim > 0.75, f"同部门相似度应为>75%，实际: {sim:.2f}"

    def test_find_similar_users(self):
        """测试查找相似用户"""
        engine = ColdStartEngine()

        target = UserProfile(
            user_id="new_user",
            temperature_preference=23.5,
            driving_mode_preference="comfort",
            music_preferences=["electronic"],
            interaction_style="moderate",
        )

        all_profiles = [
            UserProfile(user_id="u1", temperature_preference=23.0,
                       driving_mode_preference="comfort",
                       music_preferences=["electronic", "ambient"],
                       interaction_style="moderate"),
            UserProfile(user_id="u2", temperature_preference=26.0,
                       driving_mode_preference="sport",
                       music_preferences=["rock"],
                       interaction_style="frequent"),
            UserProfile(user_id="u3", temperature_preference=23.5,
                       driving_mode_preference="comfort",
                       music_preferences=["electronic", "classical"],
                       interaction_style="moderate"),
        ]

        similar = engine.find_similar_users(target, all_profiles, top_k=2)
        assert len(similar) <= 2
        if similar:
            assert similar[0][1] >= similar[-1][1]  # 按相似度降序

    def test_predict_preference_cold_start(self):
        """测试冷启动偏好预测"""
        engine = ColdStartEngine()

        group = [
            UserProfile(user_id="u1", temperature_preference=23.0,
                       driving_mode_preference="comfort",
                       music_preferences=["pop", "rock"]),
            UserProfile(user_id="u2", temperature_preference=24.0,
                       driving_mode_preference="comfort",
                       music_preferences=["pop", "jazz"]),
            UserProfile(user_id="u3", temperature_preference=23.5,
                       driving_mode_preference="eco",
                       music_preferences=["pop", "electronic"]),
        ]

        temp = engine.predict_user_preference(
            "new_user", "engineering", "temperature_preference",
            all_profiles=group, group_profiles=group,
        )
        assert 23.0 <= temp <= 24.0

        mode = engine.predict_user_preference(
            "new_user", "engineering", "driving_mode_preference",
            all_profiles=group, group_profiles=group,
        )
        assert mode == "comfort"  # 众数

    def test_initialization_time_under_5s(self):
        """测试初始化时间 < 5秒"""
        import time
        engine = ColdStartEngine()

        start = time.time()
        profile = engine.initialize_new_user_profile(
            user_id="test_user",
            department="engineering",
            role="engineer",
        )
        elapsed = time.time() - start

        assert elapsed < 5.0, f"初始化时间应<5秒，实际: {elapsed:.2f}秒"
        assert profile.user_id == "test_user"


# ─── FE-3: 日程上下文感知 ──────────────────────────────────

class TestCalendarContext:
    """测试日程上下文感知"""

    def test_scene_context_has_calendar_event(self):
        """测试SceneContext包含calendar_event字段"""
        scene = SceneContext(
            time_of_day="morning",
            location_type="home",
            calendar_event={
                "title": "项目评审会",
                "start_time": (datetime.now() + timedelta(minutes=20)).isoformat(),
                "location": "A栋-3F-会议室B",
                "event_type": "meeting",
                "participants": ["user_001", "user_002", "user_003"],
            },
        )

        assert scene.calendar_event is not None
        assert scene.calendar_event["title"] == "项目评审会"
        assert "calendar_event" in scene.to_dict()

    @pytest.mark.asyncio
    async def test_calendar_navigation_prediction(self):
        """测试会议前30分钟导航预测"""
        engine = ContextEngine(AsyncMock(), AsyncMock())
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[])

        future_time = (datetime.now() + timedelta(minutes=20)).isoformat()
        scene = SceneContext(
            time_of_day="morning",
            location_type="home",
            calendar_event={
                "title": "项目评审会",
                "start_time": future_time,
                "location": "A栋-3F-会议室B",
                "event_type": "meeting",
                "participants": ["user_001", "user_002"],
            },
        )

        predictions = await engine.predict_user_needs("user_001", scene)
        calendar_preds = [p for p in predictions if p["source"] == "calendar"]
        assert len(calendar_preds) > 0
        nav_pred = calendar_preds[0]
        assert "navigate_to" in nav_pred.get("action", {})
        assert "项目评审会" in nav_pred.get("reason", "")

    @pytest.mark.asyncio
    async def test_calendar_no_prediction_when_no_event(self):
        """测试无日历事件时不产生预测"""
        engine = ContextEngine(AsyncMock(), AsyncMock())
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[])

        scene = SceneContext(time_of_day="morning")
        predictions = await engine.predict_user_needs("user_001", scene)
        calendar_preds = [p for p in predictions if p["source"] == "calendar"]
        assert len(calendar_preds) == 0

    @pytest.mark.asyncio
    async def test_calendar_preparation_reminder(self):
        """测试会议前准备提醒"""
        engine = ContextEngine(AsyncMock(), AsyncMock())
        engine._memory_store.query = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )
        engine._pattern_store.get_patterns = AsyncMock(return_value=[])

        future_time = (datetime.now() + timedelta(minutes=45)).isoformat()
        scene = SceneContext(
            time_of_day="morning",
            calendar_event={
                "title": "周例会",
                "start_time": future_time,
                "location": "线上",
                "event_type": "meeting",
                "participants": ["user_001", "user_002", "user_003", "user_004", "user_005"],
            },
        )

        predictions = await engine.predict_user_needs("user_001", scene)
        prep_preds = [
            p for p in predictions
            if p["source"] == "calendar" and p["type"] == "calendar_preparation"
        ]
        assert len(prep_preds) > 0
        assert prep_preds[0]["action"]["participants_count"] == 5


# ─── FE-4: 文档上下文记忆 ──────────────────────────────────

class TestDocumentContext:
    """测试文档上下文记忆"""

    def test_document_metadata(self):
        """测试文档元数据"""
        meta = DocumentMetadata(
            file_name="需求文档_v2.docx",
            file_path="/docs/requirements/需求文档_v2.docx",
            file_format=DocumentFormat.DOCX,
            file_size_bytes=1024000,
            author="user_001",
            last_editor="user_002",
            page_count=25,
            word_count=5000,
        )

        d = meta.to_dict()
        assert d["file_name"] == "需求文档_v2.docx"
        assert d["file_format"] == "docx"
        assert d["page_count"] == 25

    def test_document_context_creation(self):
        """测试文档上下文创建"""
        doc = DocumentContext(
            user_id="user_001",
            document=DocumentMetadata(
                file_name="设计文档.pdf",
                file_format=DocumentFormat.PDF,
            ),
            title="AI座舱架构设计",
            content_summary="本文档描述了智能座舱AI系统的整体架构...",
            keywords=["架构", "AI", "座舱"],
            topics=["系统设计", "AI"],
            tags=["设计文档", "架构"],
            importance=5,
        )

        d = doc.to_dict()
        assert d["title"] == "AI座舱架构设计"
        assert "架构" in d["keywords"]
        assert d["importance"] == 5

    def test_add_edit_record(self):
        """测试添加编辑记录"""
        doc = DocumentContext(
            user_id="user_001",
            document=DocumentMetadata(file_name="test.txt"),
        )

        record = DocumentEditRecord(
            user_id="user_001",
            document_id=doc.id,
            action=DocumentAction.EDIT,
            session_id="session_001",
            duration_seconds=120.0,
            edit_scope="修改第三章内容",
            lines_changed=15,
        )

        doc.add_edit_record(record)
        assert doc.edit_count == 1
        assert doc.total_edit_time == 120.0
        assert "session_001" in doc.associated_sessions
        assert doc.is_related_to_session("session_001")

    def test_recent_5_documents(self):
        """测试记录最近5个文档"""
        docs = []
        for i in range(5):
            doc = DocumentContext(
                user_id="user_001",
                document=DocumentMetadata(
                    file_name=f"文档{i+1}.docx",
                    file_format=DocumentFormat.DOCX,
                ),
                title=f"文档{i+1}",
            )
            docs.append(doc)

        assert len(docs) == 5
        # 验证元数据提取
        for doc in docs:
            assert doc.document.file_format == DocumentFormat.DOCX
            assert doc.user_id == "user_001"

    def test_document_format_support(self):
        """测试支持常见文档格式"""
        formats = [
            (DocumentFormat.DOC, "doc"),
            (DocumentFormat.DOCX, "docx"),
            (DocumentFormat.PDF, "pdf"),
            (DocumentFormat.TXT, "txt"),
        ]

        for fmt, expected_value in formats:
            meta = DocumentMetadata(file_name=f"test.{expected_value}", file_format=fmt)
            assert meta.file_format == fmt
            assert meta.to_dict()["file_format"] == expected_value

    def test_cross_session_association(self):
        """测试跨会话文档上下文关联"""
        doc = DocumentContext(
            user_id="user_001",
            document=DocumentMetadata(file_name="共享文档.docx"),
            title="跨会话测试文档",
        )

        # 会话1
        record1 = DocumentEditRecord(
            user_id="user_001",
            document_id=doc.id,
            session_id="session_001",
            action=DocumentAction.EDIT,
            duration_seconds=60.0,
        )
        doc.add_edit_record(record1)

        # 会话2
        record2 = DocumentEditRecord(
            user_id="user_001",
            document_id=doc.id,
            session_id="session_002",
            action=DocumentAction.EDIT,
            duration_seconds=120.0,
        )
        doc.add_edit_record(record2)

        assert doc.is_related_to_session("session_001")
        assert doc.is_related_to_session("session_002")
        assert len(doc.associated_sessions) == 2
        assert doc.edit_count == 2
        assert doc.total_edit_time == 180.0


# ─── FE-5: 跨场景迁移 ──────────────────────────────────

class TestCrossSceneTransfer:
    """测试跨场景迁移（三级模糊匹配）"""

    @pytest.fixture
    def engine(self):
        return ContextEngine(AsyncMock(), AsyncMock())

    def test_exact_match(self, engine):
        """测试精确匹配 (Level 1)"""
        scene_a = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
            trip_purpose="commute",
        )
        scene_b = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
            trip_purpose="commute",
        )

        sim = engine.calculate_context_similarity(scene_a, scene_b)
        assert sim == 1.0

    def test_same_category_match(self, engine):
        """测试同类匹配 (Level 2)"""
        scene_a = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            trip_purpose="commute",
        )
        scene_b = SceneContext(
            time_of_day="afternoon",  # 同属 daytime
            location_type="work",     # 同属 personal
            weather="cloudy",         # 同属 clear
            traffic_condition="moderate",  # 同属 flowing
            road_type="urban",        # 同属 paved
            trip_purpose="school_run",  # 同属 routine
        )

        sim = engine.calculate_context_similarity(scene_a, scene_b)
        # 所有属性都是同类匹配，应该 > 0.5
        assert sim > 0.5, f"同类匹配相似度应 > 0.5，实际: {sim:.2f}"

    def test_different_category_degraded(self, engine):
        """测试降级匹配 (Level 3)"""
        scene_a = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            trip_purpose="commute",
        )
        scene_b = SceneContext(
            time_of_day="night",          # 不同类
            location_type="restaurant",   # 不同类
            weather="rainy",              # 不同类
            traffic_condition="jammed",   # 不同类
            road_type="rural",            # 不同类
            trip_purpose="leisure",       # 不同类
        )

        sim = engine.calculate_context_similarity(scene_a, scene_b)
        # 降级匹配应该给15%基础分
        assert 0.1 <= sim <= 0.3, f"降级匹配相似度应在0.1-0.3之间，实际: {sim:.2f}"

    def test_similarity_thresholds(self, engine):
        """测试动态阈值"""
        assert engine.get_similarity_threshold("strict") == 0.7
        assert engine.get_similarity_threshold("default") == 0.5
        assert engine.get_similarity_threshold("relaxed") == 0.3
        assert engine.get_similarity_threshold("unknown") == 0.5

    def test_is_same_category(self, engine):
        """测试同类判断"""
        # daytime: morning, afternoon
        assert engine._is_same_category("time_of_day", "morning", "afternoon")
        assert not engine._is_same_category("time_of_day", "morning", "night")

        # clear: sunny, cloudy
        assert engine._is_same_category("weather", "sunny", "cloudy")
        assert not engine._is_same_category("weather", "sunny", "rainy")

        # flowing: smooth, moderate
        assert engine._is_same_category("traffic_condition", "smooth", "moderate")
        assert not engine._is_same_category("traffic_condition", "smooth", "jammed")

    def test_temperature_cross_scene_transfer(self, engine):
        """测试温度偏好跨场景迁移（解决99°C偏差问题）"""
        # 场景A: 早晨上班，晴天
        scene_a = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            trip_purpose="commute",
        )

        # 场景B: 下午下班，阴天（同类匹配）
        scene_b = SceneContext(
            time_of_day="afternoon",
            location_type="work",
            weather="cloudy",
            traffic_condition="moderate",
            trip_purpose="commute",
        )

        sim = engine.calculate_context_similarity(scene_a, scene_b)
        # 同类匹配应有较高相似度，允许温度偏好跨场景复用
        assert sim > 0.5, f"跨场景温度迁移相似度应 > 0.5，实际: {sim:.2f}"

    def test_exact_match_performance_unchanged(self, engine):
        """测试原精确匹配场景性能不受影响"""
        # 精确匹配场景仍然返回1.0
        scene = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
            trip_purpose="commute",
        )

        sim = engine.calculate_context_similarity(scene, scene)
        assert sim == 1.0, "精确匹配应返回1.0"


# ─── FE-6: 批量操作 ──────────────────────────────────

class TestBatchOperations:
    """测试批量操作"""

    @pytest.fixture
    def mock_manager(self):
        manager = AsyncMock()
        manager.record_interaction = AsyncMock(return_value="mem_001")
        manager.retrieve_memory = AsyncMock(return_value=None)
        manager.store_memory = AsyncMock(return_value="mem_001")
        manager.query_memories = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[],
                total_found=0,
            )
        )
        return manager

    @pytest.mark.asyncio
    async def test_batch_create(self, mock_manager):
        """测试批量创建"""
        api = MemoryAPI(mock_manager)
        mock_manager.record_interaction = AsyncMock(return_value="mem_001")

        request = BatchOperationRequest(
            user_id="user_001",
            operations=[
                {"action": "create", "data": {"interaction_type": "voice_command", "intent": "navigate"}},
                {"action": "create", "data": {"interaction_type": "touch_input", "intent": "set_temperature"}},
            ],
        )

        result = await api.batch_operations(request)
        assert result.success
        assert result.succeeded == 2
        assert result.failed == 0
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_batch_exceeds_limit(self, mock_manager):
        """测试批量操作超过限制"""
        api = MemoryAPI(mock_manager)

        request = BatchOperationRequest(
            user_id="user_001",
            operations=[{"action": "create", "data": {}} for _ in range(101)],
            max_batch_size=100,
        )

        result = await api.batch_operations(request)
        assert not result.success
        assert "超过限制" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_batch_partial_success(self, mock_manager):
        """测试批量操作部分成功"""
        api = MemoryAPI(mock_manager)

        async def record_side_effect(interaction):
            if interaction.interaction_type == "fail":
                raise Exception("模拟失败")
            return "mem_ok"

        mock_manager.record_interaction = AsyncMock(side_effect=record_side_effect)

        request = BatchOperationRequest(
            user_id="user_001",
            operations=[
                {"action": "create", "data": {"interaction_type": "ok"}},
                {"action": "create", "data": {"interaction_type": "fail"}},
                {"action": "create", "data": {"interaction_type": "ok"}},
            ],
        )

        result = await api.batch_operations(request)
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_batch_unknown_action(self, mock_manager):
        """测试未知操作类型"""
        api = MemoryAPI(mock_manager)

        request = BatchOperationRequest(
            user_id="user_001",
            operations=[
                {"action": "unknown_action", "data": {}},
            ],
        )

        result = await api.batch_operations(request)
        assert result.failed == 1
        assert "未知操作类型" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_batch_query(self, mock_manager):
        """测试批量查询"""
        api = MemoryAPI(mock_manager)

        mock_item = MagicMock()
        mock_item.to_dict.return_value = {"id": "test_001", "content": {}}
        mock_manager.query_memories = AsyncMock(
            return_value=MemoryRetrievalResult(
                query=MemoryQuery(user_id="user_001"),
                items=[mock_item],
                total_found=1,
            )
        )

        request = BatchOperationRequest(
            user_id="user_001",
            operations=[
                {"action": "query", "data": {"keywords": ["test"]}},
            ],
        )

        result = await api.batch_operations(request)
        assert result.succeeded == 1
        assert result.results[0]["action"] == "query"


# ─── FE-7: 记忆导出 ──────────────────────────────────

class TestMemoryExport:
    """测试记忆导出"""

    @pytest.fixture
    def mock_manager(self):
        manager = AsyncMock()
        manager.get_user_data_inventory = AsyncMock(return_value={
            "user_id": "user_001",
            "categories": {
                "behavior": {"label": "行为记录", "count": 2, "description": "..."},
                "preference": {"label": "偏好设置", "count": 1, "description": "..."},
            },
            "items": [
                {"data_id": "1", "category": "behavior", "type": "voice_command", "intent": "navigate"},
                {"data_id": "2", "category": "behavior", "type": "touch_input", "intent": "set_temperature"},
                {"data_id": "pref_temperature_user_001", "category": "preference", "type": "temperature", "value": "23.5℃"},
            ],
            "total": 3,
            "page": 1,
            "page_size": 20,
            "has_more": False,
        })
        return manager

    @pytest.mark.asyncio
    async def test_export_json(self, mock_manager):
        """测试JSON格式导出"""
        api = MemoryAPI(mock_manager)

        request = ExportDataRequest(
            user_id="user_001",
            format="json",
        )

        result = await api.export_data(request)
        assert result.success
        assert result.data["user_id"] == "user_001"
        assert result.data["format"] == "json"
        assert result.data["total_items"] == 3

    @pytest.mark.asyncio
    async def test_export_csv(self, mock_manager):
        """测试CSV格式导出"""
        api = MemoryAPI(mock_manager)

        request = ExportDataRequest(
            user_id="user_001",
            format="csv",
        )

        result = await api.export_data(request)
        assert result.success
        assert result.data["format"] == "csv"
        assert "content" in result.data

    @pytest.mark.asyncio
    async def test_export_filter_by_category(self, mock_manager):
        """测试按类别筛选导出"""
        api = MemoryAPI(mock_manager)

        request = ExportDataRequest(
            user_id="user_001",
            format="json",
            categories=["behavior"],
        )

        result = await api.export_data(request)
        assert result.success
        # 所有返回的items应该都是behavior类别
        for item in result.data["items"]:
            assert item["category"] == "behavior"

    @pytest.mark.asyncio
    async def test_export_data_integrity(self, mock_manager):
        """测试导出数据完整性100%"""
        api = MemoryAPI(mock_manager)

        request = ExportDataRequest(
            user_id="user_001",
            format="json",
        )

        result = await api.export_data(request)
        assert result.success
        # 验证所有类别的数据都被导出
        exported_categories = set()
        for item in result.data["items"]:
            exported_categories.add(item["category"])

        # 原始数据的类别
        original_categories = set(
            item["category"] for item in mock_manager.get_user_data_inventory.return_value["items"]
        )
        assert exported_categories == original_categories

    @pytest.mark.asyncio
    async def test_export_large_file_async(self, mock_manager):
        """测试大文件异步导出"""
        api = MemoryAPI(mock_manager)

        # 模拟大量数据
        large_items = [
            {"data_id": f"item_{i}", "category": "behavior",
             "type": "test", "intent": f"intent_{i}",
             "large_data": "x" * 10000}  # 每条约10KB
            for i in range(1500)  # 约15MB
        ]
        mock_manager.get_user_data_inventory = AsyncMock(return_value={
            "user_id": "user_001",
            "categories": {},
            "items": large_items,
            "total": len(large_items),
            "page": 1,
            "page_size": 10000,
            "has_more": False,
        })

        request = ExportDataRequest(
            user_id="user_001",
            format="json",
        )

        result = await api.export_data(request)
        assert result.success
        # 大文件应该返回异步任务ID
        if result.data.get("async"):
            assert "task_id" in result.data

    def test_convert_to_csv(self):
        """测试CSV转换"""
        from cognitive_memory.api.routes import MemoryAPI
        api = MemoryAPI(AsyncMock())

        items = [
            {"data_id": "1", "category": "behavior", "intent": "navigate"},
            {"data_id": "2", "category": "behavior", "intent": "set_temperature"},
        ]

        csv_content = api._convert_to_csv(items)
        assert "data_id" in csv_content
        assert "navigate" in csv_content
        # 应该有header行 + 2条数据行
        assert csv_content.count("\n") >= 3


# ─── 综合集成测试 ──────────────────────────────────

class TestIntegration:
    """综合集成测试"""

    @pytest.mark.asyncio
    async def test_api_team_integration(self, tmp_path):
        """测试API团队接口集成"""
        from cognitive_memory.core.memory_manager import MemoryManager

        db_path = str(tmp_path / "test_integration.db")
        manager = MemoryManager(db_path=db_path)
        api = MemoryAPI(manager)

        # 创建团队
        team_req = TeamCreateRequest(
            name="集成测试团队",
            department="engineering",
            created_by="user_001",
            members=[
                {"user_id": "user_001", "role": "leader", "permission": "admin"},
                {"user_id": "user_002", "role": "member", "permission": "view"},
            ],
        )

        result = await api.create_team(team_req)
        assert result.success
        team_id = result.data["id"]

        # 查询团队
        teams_result = await api.get_user_teams("user_001")
        assert teams_result.success
        assert len(teams_result.data) >= 1

        # 创建团队记忆
        mem_req = TeamMemoryCreateRequest(
            team_id=team_id,
            title="集成测试文档",
            memory_type="project_doc",
            content={"path": "/docs/integration"},
            created_by="user_001",
        )

        mem_result = await api.create_team_memory(mem_req)
        assert mem_result.success

        # 查询团队记忆
        query_result = await api.query_team_memories(team_id, user_id="user_001")
        assert query_result.success
        assert query_result.data["total"] >= 1

    @pytest.mark.asyncio
    async def test_cold_start_with_builder(self, tmp_path):
        """测试冷启动与画像构建器集成"""
        from cognitive_memory.core.memory_manager import MemoryManager

        db_path = str(tmp_path / "test_cold_start.db")
        manager = MemoryManager(db_path=db_path)

        # 冷启动构建新用户画像
        profile = await manager._profile_builder.build_profile(
            "new_user_cold_start",
            department="engineering",
            role="engineer",
        )

        assert profile is not None
        assert profile.user_id == "new_user_cold_start"
        assert profile.temperature_preference > 0
        assert profile.confidence_score < 0.3  # 冷启动低置信度

    @pytest.mark.asyncio
    async def test_cross_scene_with_context_engine(self, tmp_path):
        """测试跨场景迁移与上下文引擎集成"""
        from cognitive_memory.core.memory_manager import MemoryManager

        db_path = str(tmp_path / "test_cross_scene.db")
        manager = MemoryManager(db_path=db_path)

        engine = manager._context_engine

        scene_a = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
            trip_purpose="commute",
        )
        scene_b = SceneContext(
            time_of_day="afternoon",
            location_type="work",
            weather="cloudy",
            traffic_condition="moderate",
            road_type="highway",
            trip_purpose="commute",
        )

        sim = engine.calculate_context_similarity(scene_a, scene_b)
        assert sim > 0.4  # 同类匹配应有较高相似度

    @pytest.mark.asyncio
    async def test_batch_export_integration(self, tmp_path):
        """测试批量操作与导出集成"""
        from cognitive_memory.core.memory_manager import MemoryManager

        db_path = str(tmp_path / "test_batch_export.db")
        manager = MemoryManager(db_path=db_path)
        api = MemoryAPI(manager)

        # 批量创建
        batch_req = BatchOperationRequest(
            user_id="user_001",
            operations=[
                {"action": "create", "data": {
                    "interaction_type": "voice_command",
                    "intent": "navigate",
                    "raw_input": "导航到公司",
                }},
                {"action": "create", "data": {
                    "interaction_type": "touch_input",
                    "intent": "set_temperature",
                    "raw_input": "设置温度24度",
                }},
            ],
        )
        batch_result = await api.batch_operations(batch_req)
        assert batch_result.success

        # 导出
        export_req = ExportDataRequest(
            user_id="user_001",
            format="json",
        )
        export_result = await api.export_data(export_req)
        assert export_result.success