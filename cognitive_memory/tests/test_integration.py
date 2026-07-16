"""
认知记忆模块 - 集成测试

测试 MemoryManager 核心编排器的完整流程。
"""

import pytest
import asyncio
from datetime import datetime

from cognitive_memory.models.memory import (
    MemoryType,
    MemoryItem,
    UserProfile,
    InteractionRecord,
    SceneContext,
    MemoryQuery,
    MemoryImportance,
)
class TestMemoryManagerIntegration:
    """MemoryManager 集成测试"""

    @pytest.mark.asyncio
    async def test_record_and_retrieve_interaction(self, memory_manager):
        """测试记录交互并检索"""
        scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            location_type="home",
            cabin_temperature=22.0,
        )

        interaction = InteractionRecord(
            user_id="user_001",
            interaction_type="voice_command",
            intent="set_temperature",
            raw_input="把温度调到24度",
            processed_input={"temperature": 24.0},
            scene_context=scene,
            system_response={"action": "set_temperature", "value": 24.0},
            was_successful=True,
            user_satisfaction=0.9,
            session_id="session_001",
        )

        memory_id = await memory_manager.record_interaction(interaction)
        assert memory_id is not None

        # 检索记忆
        retrieved = await memory_manager.retrieve_memory(memory_id)
        assert retrieved is not None
        assert retrieved.user_id == "user_001"
        assert "set_temperature" in retrieved.tags

    @pytest.mark.asyncio
    async def test_query_memories_multiple(self, memory_manager):
        """测试多项记忆查询"""
        scene = SceneContext(
            time_of_day="evening",
            weather="rainy",
            traffic_condition="heavy",
            road_type="urban",
        )

        # 记录多条交互
        for i in range(5):
            interaction = InteractionRecord(
                user_id="user_001",
                interaction_type="voice_command",
                intent=f"action_{i}",
                raw_input=f"执行操作{i}",
                processed_input={"seq": i},
                scene_context=scene,
                session_id="session_test",
            )
            await memory_manager.record_interaction(interaction)

        # 等待异步学习完成
        await asyncio.sleep(0.1)

        result = await memory_manager.query_memories(
            user_id="user_001",
            max_results=10,
        )
        assert result.total_found >= 5

    @pytest.mark.asyncio
    async def test_build_user_profile(self, memory_manager):
        """测试用户画像构建"""
        # 先记录一些交互
        for temp in [23.0, 24.0, 23.5, 24.0, 23.0]:
            scene = SceneContext(
                time_of_day="morning",
                weather="sunny",
                traffic_condition="smooth",
                road_type="urban",
            )
            interaction = InteractionRecord(
                user_id="user_001",
                interaction_type="voice_command",
                intent="set_temperature",
                raw_input=f"温度调到{temp}度",
                processed_input={"temperature": temp},
                scene_context=scene,
                session_id="profile_test",
            )
            await memory_manager.record_interaction(interaction)

        await asyncio.sleep(0.1)

        profile = await memory_manager.build_user_profile("user_001")
        assert profile is not None
        assert profile.user_id == "user_001"
        assert 23.0 <= profile.temperature_preference <= 24.0

    @pytest.mark.asyncio
    async def test_predict_user_needs(self, memory_manager):
        """测试用户需求预测"""
        # 记录路线偏好模式
        morning_scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            location_type="home",
        )

        for _ in range(4):
            interaction = InteractionRecord(
                user_id="user_001",
                interaction_type="voice_command",
                intent="navigate",
                raw_input="导航到公司",
                processed_input={
                    "destination": "公司",
                    "trip_purpose": "commute",
                },
                scene_context=morning_scene,
                session_id="route_test",
            )
            await memory_manager.record_interaction(interaction)

        await asyncio.sleep(0.2)

        # 预测当前场景的需求
        current_scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="moderate",
            road_type="highway",
            location_type="home",
        )

        predictions = await memory_manager.predict_user_needs(
            "user_001", current_scene
        )
        assert len(predictions) > 0

    @pytest.mark.asyncio
    async def test_scene_change_detection(self, memory_manager):
        """测试场景变化检测"""
        previous = SceneContext(
            time_of_day="morning",
            weather="sunny",
            engine_status="idle",
            traffic_condition="smooth",
            road_type="urban",
        )
        current = SceneContext(
            time_of_day="morning",
            weather="rainy",
            engine_status="driving",
            traffic_condition="heavy",
            road_type="highway",
        )

        result = await memory_manager.detect_scene_change(previous, current)
        assert result["has_changed"] is True
        assert "weather" in result["changes"]
        assert "engine_status" in result["changes"]

    @pytest.mark.asyncio
    async def test_forget_user_data(self, memory_manager):
        """测试用户数据删除"""
        interaction = InteractionRecord(
            user_id="user_002",
            interaction_type="voice_command",
            intent="test",
            raw_input="test",
            processed_input={},
            session_id="forget_test",
        )
        await memory_manager.record_interaction(interaction)

        count = await memory_manager.forget_user_data("user_002")
        assert count > 0

        # 确认已删除
        result = await memory_manager.query_memories(user_id="user_002")
        assert result.total_found == 0

    @pytest.mark.asyncio
    async def test_context_aware_memories(self, memory_manager):
        """测试上下文感知记忆检索"""
        morning_scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            location_type="home",
        )

        evening_scene = SceneContext(
            time_of_day="evening",
            weather="rainy",
            traffic_condition="heavy",
            road_type="urban",
            location_type="work",
        )

        # 记录早晨场景的交互
        for _ in range(3):
            interaction = InteractionRecord(
                user_id="user_001",
                interaction_type="voice_command",
                intent="play_music",
                raw_input="播放轻音乐",
                processed_input={"genre": "轻音乐"},
                scene_context=morning_scene,
                session_id="ctx_test",
            )
            await memory_manager.record_interaction(interaction)

        await asyncio.sleep(0.1)

        # 在早晨场景中检索
        result = await memory_manager.get_context_aware_memories(
            "user_001", morning_scene
        )
        assert result.total_found > 0

    @pytest.mark.asyncio
    async def test_maintenance_cycle(self, memory_manager):
        """测试维护周期"""
        # 启动维护
        await memory_manager.start_maintenance(interval_seconds=1)

        # 记录一些记忆
        interaction = InteractionRecord(
            user_id="user_001",
            interaction_type="voice_command",
            intent="test",
            raw_input="test",
            processed_input={},
            session_id="maintenance_test",
        )
        await memory_manager.record_interaction(interaction)

        # 等待维护周期
        await asyncio.sleep(1.5)

        stats = await memory_manager.get_stats()
        assert "short_term_memories" in stats

        await memory_manager.stop_maintenance()