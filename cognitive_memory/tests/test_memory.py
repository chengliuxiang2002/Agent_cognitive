"""
认知记忆模块 - 单元测试
"""

import pytest
from datetime import datetime

from cognitive_memory.models.memory import (
    MemoryType,
    MemoryItem,
    UserProfile,
    InteractionRecord,
    SceneContext,
    BehaviorPattern,
    MemoryQuery,
    MemoryImportance,
)
from cognitive_memory.storage import ShortTermMemoryStore
from cognitive_memory.learner import MemoryDecayEngine, MemoryEncoder, PatternLearner


class TestMemoryItem:
    """记忆条目测试"""

    def test_create_memory_item(self):
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
        )
        assert item.id is not None
        assert item.user_id == "user_001"
        assert item.strength == 1.0
        assert item.importance == MemoryImportance.MEDIUM

    def test_to_dict(self):
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={"key": "value"},
            tags=["test", "demo"],
        )
        d = item.to_dict()
        assert d["user_id"] == "user_001"
        assert d["memory_type"] == "short_term"
        assert d["content"]["key"] == "value"
        assert "test" in d["tags"]


class TestShortTermMemoryStore:
    """短期记忆存储测试"""

    async def test_store_and_retrieve(self):
        store = ShortTermMemoryStore(max_capacity=10)
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={"greeting": "hello"},
        )

        await store.store(item)
        retrieved = await store.retrieve(item.id)

        assert retrieved is not None
        assert retrieved.user_id == "user_001"
        assert retrieved.content["greeting"] == "hello"

    async def test_capacity_limit(self):
        store = ShortTermMemoryStore(max_capacity=3)
        for i in range(5):
            item = MemoryItem(
                user_id="user_001",
                memory_type=MemoryType.SHORT_TERM,
                content={"index": i},
            )
            await store.store(item)

        assert store.size == 3

    async def test_query_by_user(self):
        store = ShortTermMemoryStore()
        for i in range(3):
            item = MemoryItem(
                user_id="user_001",
                memory_type=MemoryType.EPISODIC,
                content={"seq": i},
            )
            await store.store(item)

        query = MemoryQuery(user_id="user_001", max_results=10)
        result = await store.query(query)

        assert result.total_found == 3

    async def test_delete_by_user(self):
        store = ShortTermMemoryStore()
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={},
        )
        await store.store(item)

        count = await store.delete_by_user("user_001")
        assert count == 1
        assert store.size == 0


class TestMemoryDecay:
    """记忆衰减引擎测试"""

    def test_decay_reduces_strength(self):
        engine = MemoryDecayEngine(base_decay_rate=0.1)
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={},
            strength=1.0,
            last_accessed_at=datetime(2020, 1, 1),  # 很久以前
        )

        decayed = engine.calculate_decay(item)
        assert decayed < 1.0

    def test_reinforcement_increases_strength(self):
        engine = MemoryDecayEngine(reinforcement_gain=0.2)
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={},
            strength=0.5,
        )

        new_strength = engine.apply_reinforcement(item)
        assert new_strength > 0.5

    def test_critical_memory_decays_slower(self):
        from datetime import timedelta

        engine = MemoryDecayEngine(base_decay_rate=0.5)
        # 使用较近的时间点，确保衰减差异在 min_strength 之上可见
        recent_past = datetime.now() - timedelta(hours=2)
        normal = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={},
            strength=1.0,
            importance=MemoryImportance.LOW,
            min_strength=0.0,
            last_accessed_at=recent_past,
        )
        critical = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.SHORT_TERM,
            content={},
            strength=1.0,
            importance=MemoryImportance.CRITICAL,
            min_strength=0.0,
            last_accessed_at=recent_past,
        )

        assert engine.calculate_decay(critical) > engine.calculate_decay(normal)


class TestMemoryEncoder:
    """记忆编码器测试"""

    def test_encode_interaction(self):
        encoder = MemoryEncoder()
        scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="smooth",
            road_type="highway",
            location_type="home",
        )
        record = InteractionRecord(
            user_id="user_001",
            intent="set_temperature",
            raw_input="温度调到23度",
            processed_input={"temperature": 23.0},
            scene_context=scene,
            was_successful=True,
            user_satisfaction=0.9,
        )

        item = encoder.encode_interaction(record, "user_001")
        assert item.user_id == "user_001"
        assert item.memory_type == MemoryType.EPISODIC
        assert "set_temperature" in item.tags
        assert "morning" in item.tags

    def test_importance_assessment(self):
        encoder = MemoryEncoder()
        safety_record = InteractionRecord(
            user_id="user_001",
            raw_input="前方有危险，请刹车",
            intent="alert",
            processed_input={},
        )
        item = encoder.encode_interaction(safety_record, "user_001")
        assert item.importance == MemoryImportance.CRITICAL


class TestPatternLearner:
    """行为模式学习器测试"""

    def test_learn_temperature_preference(self):
        learner = PatternLearner(min_occurrences=2)
        profile = UserProfile(user_id="user_001")

        scene = SceneContext(
            time_of_day="morning",
            weather="cold",
            traffic_condition="smooth",
            road_type="urban",
        )

        interactions = []
        for _ in range(3):
            interactions.append(InteractionRecord(
                user_id="user_001",
                intent="set_temperature",
                processed_input={"temperature": 24.0},
                scene_context=scene,
            ))

        patterns = learner.learn_temperature_preference(interactions, profile)
        assert len(patterns) > 0
        assert patterns[0].pattern_type == "temperature"
        assert "set_temperature" in patterns[0].expected_action

    def test_no_patterns_with_insufficient_data(self):
        learner = PatternLearner(min_occurrences=5)
        profile = UserProfile(user_id="user_001")

        scene = SceneContext(
            time_of_day="morning",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
        )

        interactions = [InteractionRecord(
            user_id="user_001",
            intent="set_temperature",
            processed_input={"temperature": 24.0},
            scene_context=scene,
        )]

        patterns = learner.learn_temperature_preference(interactions, profile)
        assert len(patterns) == 0


class TestSceneContext:
    """场景上下文测试"""

    def test_context_key_generation(self):
        scene = SceneContext(
            time_of_day="morning",
            location_type="home",
            weather="sunny",
            traffic_condition="smooth",
            road_type="urban",
        )
        key = scene.get_context_key()
        assert "morning" in key
        assert "home" in key
        assert "sunny" in key

    def test_to_dict(self):
        scene = SceneContext(
            time_of_day="evening",
            weather="rainy",
            vehicle_speed=60.0,
            cabin_temperature=22.0,
        )
        d = scene.to_dict()
        assert d["time_of_day"] == "evening"
        assert d["weather"] == "rainy"
        assert d["vehicle_speed"] == 60.0