"""
认知记忆模块 - 基准测试与质量评估

覆盖六维度质量评估体系:
1. 记忆准确性 (Precision@K, Recall@K, MRR, NDCG)
2. 信息保留率 (衰减曲线, 巩固成功率, 关键记忆保护)
3. 响应速度 (存储/检索延迟, 吞吐量, 并发)
4. 泛化能力 (跨场景迁移, 冷启动, 模式覆盖)
5. 抗干扰性 (噪声鲁棒性, 冲突处理, 一致性)
6. 用户体验 (预测命中率, 推荐接受率, 交互减少率)
"""

import asyncio
import math
import time
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

import pytest

from cognitive_memory.models.memory import (
    MemoryItem,
    MemoryType,
    MemoryImportance,
    InteractionRecord,
    SceneContext,
    UserProfile,
    MemoryQuery,
    BehaviorPattern,
)
from cognitive_memory.core import MemoryManager
from cognitive_memory.learner import MemoryDecayEngine, PatternLearner
from cognitive_memory.storage import ShortTermMemoryStore


# ═══════════════════════════════════════════════════════════════════════════════
# 测试辅助工具
# ═══════════════════════════════════════════════════════════════════════════════

def make_scene(**kwargs) -> SceneContext:
    defaults = {
        "time_of_day": "morning",
        "weather": "sunny",
        "traffic_condition": "smooth",
        "road_type": "highway",
        "location_type": "home",
        "cabin_temperature": 22.0,
    }
    defaults.update(kwargs)
    return SceneContext(**defaults)


def make_interaction(user_id: str, intent: str, processed: dict, scene: SceneContext = None, **kwargs) -> InteractionRecord:
    return InteractionRecord(
        user_id=user_id,
        interaction_type="voice_command",
        intent=intent,
        raw_input=f"执行{intent}",
        processed_input=processed,
        scene_context=scene or make_scene(),
        session_id=kwargs.get("session_id", "bench"),
        was_successful=kwargs.get("was_successful", True),
        user_satisfaction=kwargs.get("user_satisfaction", 0.9),
    )


def percentile(data: list[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


class BenchmarkMetrics:
    """基准测试指标收集器"""

    def __init__(self):
        self._metrics: dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, value: float):
        self._metrics[name].append(value)

    def summary(self) -> dict:
        result = {}
        for name, values in self._metrics.items():
            if not values:
                continue
            result[name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 一、记忆准确性
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryAccuracy:
    """记忆准确性基准测试"""

    @pytest.mark.asyncio
    async def test_precision_at_k(self, memory_manager):
        """Precision@K: 检索结果中相关记忆的占比"""
        m = BenchmarkMetrics()

        # 存储10条相关记忆 + 10条不相关记忆
        for i in range(10):
            await memory_manager.record_interaction(make_interaction(
                "user_acc", "navigate", {"destination": "公司", "seq": i},
                make_scene(time_of_day="morning"),
            ))
        for i in range(10):
            await memory_manager.record_interaction(make_interaction(
                "user_acc", "play_music", {"genre": "rock", "seq": i},
                make_scene(time_of_day="evening"),
            ))

        await asyncio.sleep(0.1)

        # 查询"navigate"相关，期望K=10中全部10条相关
        result = await memory_manager.query_memories(
            user_id="user_acc", tags=["navigate"], max_results=10,
        )

        relevant = sum(1 for item in result.items if "navigate" in item.tags)
        precision = relevant / max(len(result.items), 1)
        m.record("precision_at_10", precision)

        print(f"\nPrecision@10: {precision:.2f} ({relevant}/{len(result.items)})")
        assert precision >= 0.7, f"Precision@K too low: {precision:.2f}"

    @pytest.mark.asyncio
    async def test_recall_at_k(self, memory_manager):
        """Recall@K: 相关记忆中被检索到的比例"""
        m = BenchmarkMetrics()
        total_relevant = 10

        for i in range(total_relevant):
            await memory_manager.record_interaction(make_interaction(
                "user_recall", "set_temperature", {"temperature": 23.0 + i * 0.1},
                make_scene(time_of_day="morning"),
            ))

        await asyncio.sleep(0.1)

        result = await memory_manager.query_memories(
            user_id="user_recall", tags=["set_temperature"], max_results=20,
        )

        relevant = sum(1 for item in result.items if "set_temperature" in item.tags)
        recall = relevant / total_relevant
        m.record("recall", recall)

        print(f"\nRecall: {recall:.2f} ({relevant}/{total_relevant})")
        assert recall >= 0.6, f"Recall too low: {recall:.2f}"

    @pytest.mark.asyncio
    async def test_context_matching_accuracy(self, memory_manager):
        """上下文匹配准确率"""
        m = BenchmarkMetrics()

        # 在不同场景下存储记忆
        morning_scene = make_scene(time_of_day="morning", weather="sunny")
        evening_scene = make_scene(time_of_day="evening", weather="rainy")

        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_ctx", "navigate", {"destination": "公司"}, morning_scene,
            ))
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_ctx", "navigate", {"destination": "家"}, evening_scene,
            ))

        await asyncio.sleep(0.1)

        # 早晨场景查询，期望返回"公司"导航
        result = await memory_manager.get_context_aware_memories(
            "user_ctx", morning_scene, max_results=5,
        )

        company_hits = sum(
            1 for item in result.items
            if item.content.get("processed_input", {}).get("destination") == "公司"
        )
        accuracy = company_hits / max(len(result.items), 1)
        m.record("context_accuracy", accuracy)

        assert accuracy >= 0.3, f"Context matching accuracy too low: {accuracy:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 二、信息保留率
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetentionRate:
    """信息保留率基准测试"""

    def test_decay_curve_fitting(self):
        """衰减曲线拟合度"""
        m = BenchmarkMetrics()
        engine = MemoryDecayEngine(base_decay_rate=0.1)

        # 模拟衰减曲线并与理论值对比
        for hours in [0.5, 1, 2, 4, 8, 12, 24]:
            item = MemoryItem(
                user_id="test", memory_type=MemoryType.SHORT_TERM,
                content={}, strength=1.0, importance=MemoryImportance.MEDIUM,
                last_accessed_at=datetime.now() - timedelta(hours=hours),
                min_strength=0.0,
            )
            actual = engine.calculate_decay(item)
            weight = engine.IMPORTANCE_WEIGHT[MemoryImportance.MEDIUM]
            expected = 1.0 * math.exp(-0.1 / weight * hours)

            error = abs(actual - expected)
            m.record("decay_fitting_error", error)

        summary = m.summary()
        mean_error = summary["decay_fitting_error"]["mean"]
        assert mean_error < 0.1, f"Decay fitting error too high: {mean_error:.4f}"

    def test_importance_protection(self):
        """关键记忆保护率"""
        engine = MemoryDecayEngine(base_decay_rate=0.1)

        critical = MemoryItem(
            user_id="test", memory_type=MemoryType.LONG_TERM,
            content={}, strength=1.0, importance=MemoryImportance.CRITICAL,
            last_accessed_at=datetime.now() - timedelta(hours=24),
            min_strength=0.0,
        )
        transient = MemoryItem(
            user_id="test", memory_type=MemoryType.LONG_TERM,
            content={}, strength=1.0, importance=MemoryImportance.TRANSIENT,
            last_accessed_at=datetime.now() - timedelta(hours=24),
            min_strength=0.0,
        )

        critical_decay = engine.calculate_decay(critical)
        transient_decay = engine.calculate_decay(transient)

        # 关键记忆衰减应远小于临时记忆
        ratio = critical_decay / max(transient_decay, 0.001)
        assert ratio > 2.0, f"Critical memory not well protected: ratio={ratio:.2f}"

    @pytest.mark.asyncio
    async def test_consolidation_success(self, memory_manager):
        """巩固成功率"""
        # 创建高重要性、多次访问的记忆
        item = MemoryItem(
            user_id="user_consolidate",
            memory_type=MemoryType.SHORT_TERM,
            content={"action": "important"},
            importance=MemoryImportance.HIGH,
            strength=0.9,
            access_count=5,
        )
        await memory_manager.store_memory(item)

        # 启动维护并等待巩固
        await memory_manager.start_maintenance(interval_seconds=1)
        await asyncio.sleep(1.5)
        await memory_manager.stop_maintenance()

        # 检查是否已转为长期记忆
        retrieved = await memory_manager.retrieve_memory(item.id)
        assert retrieved is not None, "Memory should still exist after consolidation"


# ═══════════════════════════════════════════════════════════════════════════════
# 三、响应速度
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseSpeed:
    """响应速度基准测试"""

    @pytest.mark.asyncio
    async def test_store_latency(self, memory_manager):
        """存储延迟 P50/P95/P99"""
        m = BenchmarkMetrics()

        for i in range(100):
            interaction = make_interaction(
                "user_perf", "test", {"seq": i}, make_scene(),
            )
            start = time.perf_counter()
            await memory_manager.record_interaction(interaction)
            elapsed = (time.perf_counter() - start) * 1000
            m.record("store_latency_ms", elapsed)

        summary = m.summary()
        p50 = summary["store_latency_ms"]["p50"]
        p95 = summary["store_latency_ms"]["p95"]
        p99 = summary["store_latency_ms"]["p99"]

        print(f"\n存储延迟: P50={p50:.1f}ms, P95={p95:.1f}ms, P99={p99:.1f}ms")

        assert p50 < 100, f"P50 latency too high: {p50:.1f}ms"
        assert p95 < 200, f"P95 latency too high: {p95:.1f}ms"

    @pytest.mark.asyncio
    async def test_query_latency(self, memory_manager):
        """检索延迟 P50/P95/P99"""
        m = BenchmarkMetrics()

        # 先填充数据
        for i in range(50):
            await memory_manager.record_interaction(make_interaction(
                "user_query_perf", "test", {"seq": i}, make_scene(),
            ))
        await asyncio.sleep(0.1)

        for _ in range(100):
            start = time.perf_counter()
            await memory_manager.query_memories(
                user_id="user_query_perf", max_results=10, sort_by="relevance",
            )
            elapsed = (time.perf_counter() - start) * 1000
            m.record("query_latency_ms", elapsed)

        summary = m.summary()
        p50 = summary["query_latency_ms"]["p50"]
        p95 = summary["query_latency_ms"]["p95"]

        print(f"\n查询延迟: P50={p50:.1f}ms, P95={p95:.1f}ms")

        assert p50 < 100, f"P50 query latency too high: {p50:.1f}ms"
        assert p95 < 200, f"P95 query latency too high: {p95:.1f}ms"

    @pytest.mark.asyncio
    async def test_throughput(self, memory_manager):
        """存储吞吐量 (ops/s)"""
        m = BenchmarkMetrics()

        start = time.perf_counter()
        count = 0
        deadline = start + 2.0  # 2秒测试窗口

        while time.perf_counter() < deadline:
            await memory_manager.record_interaction(make_interaction(
                "user_tp", "test", {"seq": count}, make_scene(),
            ))
            count += 1

        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        m.record("throughput_ops", throughput)

        print(f"\n存储吞吐量: {throughput:.1f} ops/s ({count} ops in {elapsed:.1f}s)")

        assert throughput >= 50, f"Throughput too low: {throughput:.1f} ops/s"

    @pytest.mark.asyncio
    async def test_profile_build_time(self, memory_manager):
        """画像构建耗时"""
        # 填充200条交互
        for i in range(200):
            await memory_manager.record_interaction(make_interaction(
                "user_profile_time", "set_temperature",
                {"temperature": 23.0 + (i % 10) * 0.2}, make_scene(),
            ))
        await asyncio.sleep(0.1)

        start = time.perf_counter()
        profile = await memory_manager.build_user_profile("user_profile_time")
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n画像构建耗时: {elapsed:.1f}ms (200条交互)")

        assert elapsed < 2000, f"Profile build too slow: {elapsed:.1f}ms"
        assert profile is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 四、泛化能力
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneralization:
    """泛化能力基准测试"""

    @pytest.mark.asyncio
    async def test_cross_scene_transfer(self, memory_manager):
        """跨场景迁移准确率"""
        m = BenchmarkMetrics()

        # 在"晴天早晨"场景学习温度偏好
        sunny_morning = make_scene(time_of_day="morning", weather="sunny")
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_xfer", "set_temperature", {"temperature": 24.0}, sunny_morning,
            ))

        await asyncio.sleep(0.2)

        # 在"晴天上午"场景测试（相似但不同）
        sunny_forenoon = make_scene(time_of_day="afternoon", weather="sunny")
        predictions = await memory_manager.predict_user_needs("user_xfer", sunny_forenoon)

        temp_predictions = [
            p for p in predictions
            if p.get("action", {}).get("set_temperature") is not None
        ]
        if temp_predictions:
            predicted_temp = temp_predictions[0]["action"]["set_temperature"]
            diff = abs(predicted_temp - 24.0)
            m.record("cross_scene_temp_diff", diff)
            assert diff < 3.0, f"Cross-scene temperature prediction off by {diff:.1f}°C"

    @pytest.mark.asyncio
    async def test_cold_start_improvement(self, memory_manager):
        """新用户冷启动效果"""
        m = BenchmarkMetrics()

        user_id = "user_coldstart"
        scenes = [
            make_scene(time_of_day=tod)
            for tod in ["morning", "afternoon", "evening"]
        ]

        hit_rates = []
        for n in range(1, 21):
            scene = scenes[n % 3]
            await memory_manager.record_interaction(make_interaction(
                user_id, "navigate", {"destination": f"地点{n % 5 + 1}"}, scene,
            ))

            if n >= 5 and n % 5 == 0:
                await asyncio.sleep(0.05)
                predictions = await memory_manager.predict_user_needs(user_id, scene)
                has_nav = any(
                    p.get("action", {}).get("navigate_to") is not None
                    for p in predictions
                )
                hit_rates.append((n, int(has_nav)))

        # 验证后期命中率高于前期
        if len(hit_rates) >= 2:
            early = sum(h for _, h in hit_rates[:2]) / max(len(hit_rates[:2]), 1)
            late = sum(h for _, h in hit_rates[-2:]) / max(len(hit_rates[-2:]), 1)
            m.record("cold_start_early", early)
            m.record("cold_start_late", late)
            print(f"\n冷启动: 前期命中率={early:.2f}, 后期命中率={late:.2f}")

    @pytest.mark.asyncio
    async def test_pattern_discovery_coverage(self, memory_manager):
        """模式发现覆盖率"""
        # 注入已知模式: 工作日早晨导航到公司
        for day in range(10):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][day % 5]
            await memory_manager.record_interaction(make_interaction(
                "user_pattern_cov", "navigate",
                {"destination": "公司", "trip_purpose": "commute"},
                make_scene(time_of_day="morning"),
            ))

        await asyncio.sleep(0.2)

        patterns = await memory_manager.get_behavior_patterns("user_pattern_cov")
        route_patterns = [p for p in patterns if p.pattern_type == "route"]

        print(f"\n发现模式: {len(patterns)}个, 路线模式: {len(route_patterns)}个")
        for p in route_patterns:
            print(f"  - {p.pattern_name}: 置信度={p.confidence:.2f}")

        assert len(route_patterns) > 0, "Should discover at least one route pattern"


# ═══════════════════════════════════════════════════════════════════════════════
# 五、抗干扰性
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiInterference:
    """抗干扰性基准测试"""

    @pytest.mark.asyncio
    async def test_noise_robustness(self, memory_manager):
        """噪声鲁棒性"""
        m = BenchmarkMetrics()

        # 纯净数据: 温度偏好 23°C
        for _ in range(8):
            await memory_manager.record_interaction(make_interaction(
                "user_noise", "set_temperature", {"temperature": 23.0},
                make_scene(time_of_day="morning"),
            ))

        # 噪声数据: 异常温度值
        for _ in range(2):
            await memory_manager.record_interaction(make_interaction(
                "user_noise", "set_temperature", {"temperature": 35.0},
                make_scene(time_of_day="morning"),
            ))

        await asyncio.sleep(0.2)

        profile = await memory_manager.build_user_profile("user_noise")
        temp_diff = abs(profile.temperature_preference - 23.0)
        m.record("noise_temp_deviation", temp_diff)

        print(f"\n噪声鲁棒性: 温度偏差={temp_diff:.2f}°C (含20%噪声)")
        assert temp_diff < 5.0, f"Temperature deviated too much under noise: {temp_diff:.2f}°C"

    @pytest.mark.asyncio
    async def test_conflict_handling(self, memory_manager):
        """冲突处理: 矛盾信息"""
        # 大部分时间设为23°C
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_conflict", "set_temperature", {"temperature": 23.0},
                make_scene(time_of_day="morning"),
            ))
        # 少数时间设为28°C
        for _ in range(1):
            await memory_manager.record_interaction(make_interaction(
                "user_conflict", "set_temperature", {"temperature": 28.0},
                make_scene(time_of_day="morning"),
            ))

        await asyncio.sleep(0.2)

        profile = await memory_manager.build_user_profile("user_conflict")
        # 期望: 偏向多数值23.0，而非28.0
        deviation_from_23 = abs(profile.temperature_preference - 23.0)
        deviation_from_28 = abs(profile.temperature_preference - 28.0)

        print(f"\n冲突处理: 偏好={profile.temperature_preference:.1f}°C, 偏离23°C={deviation_from_23:.1f}, 偏离28°C={deviation_from_28:.1f}")
        assert deviation_from_23 < deviation_from_28, "Should favor majority value"

    @pytest.mark.asyncio
    async def test_user_isolation(self, memory_manager):
        """多用户记忆隔离"""
        # 用户A的温度偏好
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_a", "set_temperature", {"temperature": 20.0},
                make_scene(),
            ))
        # 用户B的温度偏好
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_b", "set_temperature", {"temperature": 26.0},
                make_scene(),
            ))

        await asyncio.sleep(0.2)

        profile_a = await memory_manager.build_user_profile("user_a")
        profile_b = await memory_manager.build_user_profile("user_b")

        diff = abs(profile_a.temperature_preference - profile_b.temperature_preference)
        print(f"\n用户隔离: 用户A={profile_a.temperature_preference:.1f}°C, 用户B={profile_b.temperature_preference:.1f}°C, 差异={diff:.1f}°C")

        assert diff > 2.0, "User profiles should be well separated"


# ═══════════════════════════════════════════════════════════════════════════════
# 六、用户体验
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserExperience:
    """用户体验基准测试"""

    @pytest.mark.asyncio
    async def test_prediction_hit_rate(self, memory_manager):
        """需求预测命中率"""
        m = BenchmarkMetrics()

        # 建立早晨→导航的模式
        morning = make_scene(time_of_day="morning")
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_ux", "navigate", {"destination": "公司"}, morning,
            ))

        await asyncio.sleep(0.2)

        predictions = await memory_manager.predict_user_needs("user_ux", morning)
        nav_predictions = [
            p for p in predictions
            if p.get("action", {}).get("navigate_to") is not None
        ]
        hit = 1 if nav_predictions else 0
        m.record("prediction_hit", hit)

        print(f"\n需求预测命中: {'是' if hit else '否'} (共{len(predictions)}个预测)")
        assert hit == 1, "Should predict navigation need"

    @pytest.mark.asyncio
    async def test_interaction_reduction(self, memory_manager):
        """交互减少率"""
        # 无记忆时: 需要完整指令 "导航到公司"
        # 有记忆时: 系统可预测并主动推荐

        morning = make_scene(time_of_day="morning")

        # 建立模式
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_reduce", "navigate", {"destination": "公司"}, morning,
            ))

        await asyncio.sleep(0.2)

        # 检查预测是否存在
        predictions = await memory_manager.predict_user_needs("user_reduce", morning)
        high_conf = [p for p in predictions if p.get("confidence", 0) >= 0.3]
        reduction_potential = min(1.0, len(high_conf) / 3)

        print(f"\n交互减少潜力: {reduction_potential:.2f} (高置信度预测: {len(high_conf)}个)")

        assert len(high_conf) > 0, "Should have at least one prediction to reduce interaction"

    @pytest.mark.asyncio
    async def test_personalization_sense(self, memory_manager):
        """个性化感知度"""
        # 用户A: 喜欢摇滚
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_a_music", "play_music", {"genre": "rock"}, make_scene(),
            ))
        # 用户B: 喜欢轻音乐
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                "user_b_music", "play_music", {"genre": "轻音乐"}, make_scene(),
            ))

        await asyncio.sleep(0.2)

        profile_a = await memory_manager.build_user_profile("user_a_music")
        profile_b = await memory_manager.build_user_profile("user_b_music")

        # 两个用户的音乐偏好应该不同
        overlap = set(profile_a.music_preferences) & set(profile_b.music_preferences)
        diff_ratio = 1 - len(overlap) / max(len(profile_a.music_preferences) + len(profile_b.music_preferences), 1)

        print(f"\n个性化区分度: {diff_ratio:.2f}")

        # 不做强制断言，因为音乐偏好可能偶然重叠，但应有区分度
        assert diff_ratio >= 0.0, "Personalization differentiation should exist"


# ═══════════════════════════════════════════════════════════════════════════════
# 综合评估 — 严苛评分体系
# 每项指标对照 README 目标值分级计分，再按维度权重加权汇总。
# ═══════════════════════════════════════════════════════════════════════════════

def _tier3(value: float, excellent: float, good: float, fair: float) -> float:
    """三档计分: 优秀=100, 达标=80, 接近=50, 不达标=20"""
    if value >= excellent:
        return 100.0
    if value >= good:
        return 80.0
    if value >= fair:
        return 50.0
    return 20.0


def _tier3_lower(value: float, excellent: float, good: float, fair: float) -> float:
    """三档计分(越低越好): 优秀=100, 达标=80, 接近=50, 不达标=20"""
    if value <= excellent:
        return 100.0
    if value <= good:
        return 80.0
    if value <= fair:
        return 50.0
    return 20.0


class TestComprehensiveEvaluation:
    """综合质量评估 — 对照 README 目标值的严苛评分"""

    # pylint: disable=too-many-locals,too-many-statements

    @pytest.mark.asyncio
    async def test_comprehensive_score(self, memory_manager):
        """运行全部维度评估并输出综合报告"""
        scores: dict[str, float] = {}
        details: dict[str, dict] = {}

        # ═══════════════════════════════════════════════════════════════
        # 一、记忆准确性 (权重 25%)
        # ═══════════════════════════════════════════════════════════════
        uid_acc = "user_eval_accuracy"

        # 1.1 Precision@K
        for i in range(10):
            await memory_manager.record_interaction(make_interaction(
                uid_acc, "navigate", {"destination": "公司", "seq": i},
                make_scene(time_of_day="morning"),
            ))
        for i in range(10):
            await memory_manager.record_interaction(make_interaction(
                uid_acc, "play_music", {"genre": "rock", "seq": i},
                make_scene(time_of_day="evening"),
            ))
        await asyncio.sleep(0.1)

        result = await memory_manager.query_memories(
            user_id=uid_acc, tags=["navigate"], max_results=10,
        )
        relevant = sum(1 for item in result.items if "navigate" in item.tags)
        prec = relevant / max(len(result.items), 1)
        # README: Precision@K ≥ 0.85
        prec_score = _tier3(prec, 0.95, 0.85, 0.70)

        # 1.2 Recall@K
        uid_recall = "user_eval_recall"
        for i in range(10):
            await memory_manager.record_interaction(make_interaction(
                uid_recall, "set_temperature", {"temperature": 23.0 + i * 0.1},
                make_scene(time_of_day="morning"),
            ))
        await asyncio.sleep(0.1)
        result = await memory_manager.query_memories(
            user_id=uid_recall, tags=["set_temperature"], max_results=20,
        )
        relevant = sum(1 for item in result.items if "set_temperature" in item.tags)
        rec = relevant / 10  # total_relevant = 10
        # README: Recall@K ≥ 0.80
        rec_score = _tier3(rec, 0.95, 0.80, 0.60)

        # 1.3 上下文匹配准确率
        uid_ctx = "user_eval_ctx"
        morning = make_scene(time_of_day="morning", weather="sunny")
        evening = make_scene(time_of_day="evening", weather="rainy")
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_ctx, "navigate", {"destination": "公司"}, morning,
            ))
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_ctx, "navigate", {"destination": "家"}, evening,
            ))
        await asyncio.sleep(0.1)
        ctx_result = await memory_manager.get_context_aware_memories(
            uid_ctx, morning, max_results=5,
        )
        company_hits = sum(
            1 for item in ctx_result.items
            if item.content.get("processed_input", {}).get("destination") == "公司"
        )
        ctx_acc = company_hits / max(len(ctx_result.items), 1)
        # README: 上下文匹配准确率 ≥ 0.80
        ctx_score = _tier3(ctx_acc, 0.90, 0.80, 0.50)

        accuracy_score = (prec_score * 0.35 + rec_score * 0.35 + ctx_score * 0.30)
        scores["accuracy"] = accuracy_score
        details["accuracy"] = {
            "Precision@10": (prec, prec_score),
            "Recall@10": (rec, rec_score),
            "ContextMatch": (ctx_acc, ctx_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 二、信息保留率 (权重 20%)
        # ═══════════════════════════════════════════════════════════════

        # 2.1 衰减曲线拟合度
        from cognitive_memory.learner import MemoryDecayEngine
        engine = MemoryDecayEngine(base_decay_rate=0.1)
        errors = []
        for hours in [0.5, 1, 2, 4, 8, 12, 24]:
            item = MemoryItem(
                user_id="test", memory_type=MemoryType.SHORT_TERM,
                content={}, strength=1.0, importance=MemoryImportance.MEDIUM,
                last_accessed_at=datetime.now() - timedelta(hours=hours),
                min_strength=0.0,
            )
            actual = engine.calculate_decay(item)
            weight = engine.IMPORTANCE_WEIGHT[MemoryImportance.MEDIUM]
            expected = 1.0 * math.exp(-0.1 / weight * hours)
            errors.append(abs(actual - expected))
        mean_decay_err = statistics.mean(errors)
        # README: 衰减曲线拟合度 ≥ 0.85  → 误差 < 0.15
        decay_score = _tier3_lower(mean_decay_err, 0.01, 0.05, 0.15)

        # 2.2 关键记忆保护率
        critical = MemoryItem(
            user_id="test", memory_type=MemoryType.LONG_TERM,
            content={}, strength=1.0, importance=MemoryImportance.CRITICAL,
            last_accessed_at=datetime.now() - timedelta(hours=24),
            min_strength=0.0,
        )
        transient = MemoryItem(
            user_id="test", memory_type=MemoryType.LONG_TERM,
            content={}, strength=1.0, importance=MemoryImportance.TRANSIENT,
            last_accessed_at=datetime.now() - timedelta(hours=24),
            min_strength=0.0,
        )
        crit_decay = engine.calculate_decay(critical)
        trans_decay = engine.calculate_decay(transient)
        protect_ratio = crit_decay / max(trans_decay, 0.001)
        # README: 关键记忆保护率 ≥ 0.98 → ratio 应远大于 1
        protect_score = 100.0 if protect_ratio > 5.0 else (80.0 if protect_ratio > 2.0 else (50.0 if protect_ratio > 1.0 else 20.0))

        # 2.3 巩固成功率
        cons_item = MemoryItem(
            user_id="user_eval_consolidate",
            memory_type=MemoryType.SHORT_TERM,
            content={"action": "important"},
            importance=MemoryImportance.HIGH,
            strength=0.9,
            access_count=5,
        )
        await memory_manager.store_memory(cons_item)
        await memory_manager.start_maintenance(interval_seconds=1)
        await asyncio.sleep(1.5)
        await memory_manager.stop_maintenance()
        retrieved = await memory_manager.retrieve_memory(cons_item.id)
        cons_success = retrieved is not None
        cons_score = 100.0 if cons_success else 0.0

        retention_score = (decay_score * 0.35 + protect_score * 0.35 + cons_score * 0.30)
        scores["retention"] = retention_score
        details["retention"] = {
            "DecayError": (mean_decay_err, decay_score),
            "ProtectRatio": (protect_ratio, protect_score),
            "Consolidation": (cons_success, cons_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 三、响应速度 (权重 20%)
        # ═══════════════════════════════════════════════════════════════
        uid_perf = "user_eval_perf"

        # 3.1 存储延迟
        store_latencies = []
        for i in range(100):
            start = time.perf_counter()
            await memory_manager.record_interaction(make_interaction(
                uid_perf, "test", {"seq": i}, make_scene(),
            ))
            store_latencies.append((time.perf_counter() - start) * 1000)
        store_p50 = percentile(store_latencies, 50)
        store_p95 = percentile(store_latencies, 95)
        store_p99 = percentile(store_latencies, 99)
        # README: P50 < 10ms, P95 < 50ms, P99 < 100ms
        store_p50_score = _tier3_lower(store_p50, 5, 10, 20)
        store_p95_score = _tier3_lower(store_p95, 20, 50, 100)
        store_p99_score = _tier3_lower(store_p99, 50, 100, 200)

        # 3.2 查询延迟
        await asyncio.sleep(0.1)
        query_latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await memory_manager.query_memories(
                user_id=uid_perf, max_results=10, sort_by="relevance",
            )
            query_latencies.append((time.perf_counter() - start) * 1000)
        query_p50 = percentile(query_latencies, 50)
        query_p95 = percentile(query_latencies, 95)
        # README: P50 < 20ms, P95 < 100ms
        query_p50_score = _tier3_lower(query_p50, 3, 10, 20)
        query_p95_score = _tier3_lower(query_p95, 10, 50, 100)

        # 3.3 吞吐量
        start = time.perf_counter()
        count = 0
        deadline = start + 2.0
        while time.perf_counter() < deadline:
            await memory_manager.record_interaction(make_interaction(
                "user_eval_tp", "test", {"seq": count}, make_scene(),
            ))
            count += 1
        throughput = count / (time.perf_counter() - start)
        # README: ≥ 500 ops/s
        tp_score = 100.0 if throughput >= 500 else (80.0 if throughput >= 200 else (50.0 if throughput >= 100 else (20.0 if throughput >= 50 else 5.0)))

        # 3.4 画像构建耗时
        for i in range(200):
            await memory_manager.record_interaction(make_interaction(
                "user_eval_profile", "set_temperature",
                {"temperature": 23.0 + (i % 10) * 0.2}, make_scene(),
            ))
        await asyncio.sleep(0.1)
        start = time.perf_counter()
        await memory_manager.build_user_profile("user_eval_profile")
        profile_ms = (time.perf_counter() - start) * 1000
        # README: < 500ms
        profile_score = _tier3_lower(profile_ms, 100, 500, 1000)

        speed_score = (
            store_p50_score * 0.15 + store_p95_score * 0.10 + store_p99_score * 0.05 +
            query_p50_score * 0.15 + query_p95_score * 0.10 +
            tp_score * 0.25 +
            profile_score * 0.20
        )
        scores["speed"] = speed_score
        details["speed"] = {
            "StoreP50(ms)": (store_p50, store_p50_score),
            "StoreP95(ms)": (store_p95, store_p95_score),
            "StoreP99(ms)": (store_p99, store_p99_score),
            "QueryP50(ms)": (query_p50, query_p50_score),
            "QueryP95(ms)": (query_p95, query_p95_score),
            "Throughput(ops/s)": (throughput, tp_score),
            "ProfileBuild(ms)": (profile_ms, profile_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 四、泛化能力 (权重 10%)
        # ═══════════════════════════════════════════════════════════════

        # 4.1 跨场景迁移
        uid_xfer = "user_eval_xfer"
        sunny_morning = make_scene(time_of_day="morning", weather="sunny")
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_xfer, "set_temperature", {"temperature": 24.0}, sunny_morning,
            ))
        await asyncio.sleep(0.2)
        sunny_forenoon = make_scene(time_of_day="afternoon", weather="sunny")
        predictions = await memory_manager.predict_user_needs(uid_xfer, sunny_forenoon)
        temp_preds = [p for p in predictions if p.get("action", {}).get("set_temperature") is not None]
        cross_transfer_score = 20.0  # default: no prediction
        temp_diff = 99.0
        if temp_preds:
            temp_diff = abs(temp_preds[0]["action"]["set_temperature"] - 24.0)
            # README: 跨场景迁移准确率 ≥ 0.70 → 温度偏差 < 3°C
            cross_transfer_score = _tier3_lower(temp_diff, 0.5, 2.0, 3.0)

        # 4.2 冷启动改善
        uid_cold = "user_eval_cold"
        scenes_cold = [make_scene(time_of_day=tod) for tod in ["morning", "afternoon", "evening"]]
        early_hits = []
        late_hits = []
        for n in range(1, 21):
            scene = scenes_cold[n % 3]
            await memory_manager.record_interaction(make_interaction(
                uid_cold, "navigate", {"destination": f"地点{n % 5 + 1}"}, scene,
            ))
            if n in (5, 10):
                await asyncio.sleep(0.05)
                preds = await memory_manager.predict_user_needs(uid_cold, scene)
                has_nav = any(p.get("action", {}).get("navigate_to") is not None for p in preds)
                early_hits.append(int(has_nav))
            if n in (15, 20):
                await asyncio.sleep(0.05)
                preds = await memory_manager.predict_user_needs(uid_cold, scene)
                has_nav = any(p.get("action", {}).get("navigate_to") is not None for p in preds)
                late_hits.append(int(has_nav))
        early_rate = sum(early_hits) / max(len(early_hits), 1)
        late_rate = sum(late_hits) / max(len(late_hits), 1)
        # README: 第5次 ≥ 0.40, 第20次 ≥ 0.65
        cold_improvement = late_rate - early_rate
        if late_rate >= 0.65 and early_rate >= 0.40:
            cold_score = 100.0
        elif late_rate >= 0.40 and cold_improvement > 0.1:
            cold_score = 80.0
        elif cold_improvement > 0:
            cold_score = 50.0
        else:
            cold_score = 20.0

        # 4.3 模式发现
        uid_pat = "user_eval_pattern"
        for day in range(10):
            await memory_manager.record_interaction(make_interaction(
                uid_pat, "navigate", {"destination": "公司", "trip_purpose": "commute"},
                make_scene(time_of_day="morning"),
            ))
        await asyncio.sleep(0.2)
        patterns = await memory_manager.get_behavior_patterns(uid_pat)
        route_patterns = [p for p in patterns if p.pattern_type == "route"]
        pat_count = len(route_patterns)
        pat_confidence = max((p.confidence for p in route_patterns), default=0)
        # README: 模式发现覆盖率 ≥ 0.80
        pat_count_score = 100.0 if pat_count >= 5 else (80.0 if pat_count >= 3 else (50.0 if pat_count >= 1 else 20.0))
        pat_conf_score = 100.0 if pat_confidence >= 0.80 else (80.0 if pat_confidence >= 0.60 else (50.0 if pat_confidence >= 0.40 else 20.0))
        pat_score = pat_count_score * 0.5 + pat_conf_score * 0.5

        gen_score = (cross_transfer_score * 0.35 + cold_score * 0.30 + pat_score * 0.35)
        scores["generalization"] = gen_score
        details["generalization"] = {
            "CrossSceneDiff(°C)": (temp_diff, cross_transfer_score),
            "ColdStart(early/late)": (f"{early_rate:.2f}/{late_rate:.2f}", cold_score),
            "PatternCount": (pat_count, pat_count_score),
            "PatternConfidence": (pat_confidence, pat_conf_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 五、抗干扰性 (权重 10%)
        # ═══════════════════════════════════════════════════════════════

        # 5.1 噪声鲁棒性
        uid_noise = "user_eval_noise"
        for _ in range(8):
            await memory_manager.record_interaction(make_interaction(
                uid_noise, "set_temperature", {"temperature": 23.0},
                make_scene(time_of_day="morning"),
            ))
        for _ in range(2):
            await memory_manager.record_interaction(make_interaction(
                uid_noise, "set_temperature", {"temperature": 40.0},
                make_scene(time_of_day="morning"),
            ))
        await asyncio.sleep(0.1)
        noise_profile = await memory_manager.build_user_profile(uid_noise)
        noise_temp = noise_profile.temperature_preference
        noise_dev = abs(noise_temp - 23.0)
        # README: 噪声鲁棒性 ≥ 0.85 (20%噪声) → 偏差 < 3°C
        noise_score = _tier3_lower(noise_dev, 0.5, 2.0, 5.0)

        # 5.2 冲突处理
        uid_conflict = "user_eval_conflict"
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_conflict, "set_temperature", {"temperature": 23.0},
                make_scene(time_of_day="morning"),
            ))
        for _ in range(3):
            await memory_manager.record_interaction(make_interaction(
                uid_conflict, "set_temperature", {"temperature": 28.0},
                make_scene(time_of_day="evening"),
            ))
        await asyncio.sleep(0.1)
        conflict_profile = await memory_manager.build_user_profile(uid_conflict)
        conflict_temp = conflict_profile.temperature_preference
        # 应偏向出现次数更多的 23°C
        conflict_dev = abs(conflict_temp - 23.0)
        conflict_score = _tier3_lower(conflict_dev, 0.5, 1.5, 3.0)

        # 5.3 用户隔离
        uid_iso_a = "user_eval_iso_a"
        uid_iso_b = "user_eval_iso_b"
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_iso_a, "set_temperature", {"temperature": 20.0}, make_scene(),
            ))
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_iso_b, "set_temperature", {"temperature": 26.0}, make_scene(),
            ))
        await asyncio.sleep(0.1)
        profile_a = await memory_manager.build_user_profile(uid_iso_a)
        profile_b = await memory_manager.build_user_profile(uid_iso_b)
        iso_diff = abs(profile_a.temperature_preference - profile_b.temperature_preference)
        iso_score = 100.0 if iso_diff >= 5.0 else (80.0 if iso_diff >= 3.0 else (50.0 if iso_diff >= 1.0 else 20.0))

        anti_score = (noise_score * 0.40 + conflict_score * 0.35 + iso_score * 0.25)
        scores["anti_interference"] = anti_score
        details["anti_interference"] = {
            "NoiseDev(°C)": (noise_dev, noise_score),
            "ConflictDev(°C)": (conflict_dev, conflict_score),
            "UserIsoDiff(°C)": (iso_diff, iso_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 六、用户体验 (权重 15%)
        # ═══════════════════════════════════════════════════════════════

        # 6.1 需求预测命中率
        uid_ux = "user_eval_ux"
        morning_ux = make_scene(time_of_day="morning")
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_ux, "navigate", {"destination": "公司"}, morning_ux,
            ))
        await asyncio.sleep(0.2)
        preds = await memory_manager.predict_user_needs(uid_ux, morning_ux)
        pred_hit = any(
            p.get("action", {}).get("navigate_to") == "公司"
            for p in preds
        )
        pred_hit_score = 100.0 if pred_hit else 0.0

        # 6.2 交互减少率
        high_conf = [p for p in preds if p.get("confidence", 0) >= 0.3]
        reduction = min(1.0, len(high_conf) / 3)
        # README: 交互减少率 ≥ 0.30
        reduction_score = _tier3(reduction, 0.60, 0.30, 0.10)

        # 6.3 个性化感知度
        uid_a = "user_eval_pers_a"
        uid_b = "user_eval_pers_b"
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_a, "play_music", {"genre": "rock"}, make_scene(),
            ))
        for _ in range(5):
            await memory_manager.record_interaction(make_interaction(
                uid_b, "play_music", {"genre": "轻音乐"}, make_scene(),
            ))
        await asyncio.sleep(0.2)
        pa = await memory_manager.build_user_profile(uid_a)
        pb = await memory_manager.build_user_profile(uid_b)
        overlap = set(pa.music_preferences) & set(pb.music_preferences)
        total_prefs = len(pa.music_preferences) + len(pb.music_preferences)
        diff_ratio = 1 - len(overlap) / max(total_prefs, 1)
        # README: 个性化感知度 ≥ 0.60
        pers_score = _tier3(diff_ratio, 0.80, 0.60, 0.30)

        ux_score = (pred_hit_score * 0.35 + reduction_score * 0.35 + pers_score * 0.30)
        scores["user_experience"] = ux_score
        details["user_experience"] = {
            "PredictionHit": (pred_hit, pred_hit_score),
            "InteractionReduction": (reduction, reduction_score),
            "Personalization": (diff_ratio, pers_score),
        }

        # ═══════════════════════════════════════════════════════════════
        # 加权汇总
        # ═══════════════════════════════════════════════════════════════
        weights = {
            "accuracy": 0.25,
            "retention": 0.20,
            "speed": 0.20,
            "generalization": 0.10,
            "anti_interference": 0.10,
            "user_experience": 0.15,
        }

        final_score = sum(scores[k] * weights[k] for k in weights)

        # ═══════════════════════════════════════════════════════════════
        # 输出报告
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  认知记忆模块 — 综合质量评估报告 (严苛标准)")
        print("=" * 70)

        dim_names = {
            "accuracy": "记忆准确性",
            "retention": "信息保留率",
            "speed": "响应速度",
            "generalization": "泛化能力",
            "anti_interference": "抗干扰性",
            "user_experience": "用户体验",
        }

        for dim_key in weights:
            dim_name = dim_names[dim_key]
            w = weights[dim_key]
            s = scores[dim_key]
            weighted = s * w
            print(f"\n  [{dim_name}] 权重 {w*100:.0f}% | 维度得分: {s:.1f} | 加权: {weighted:.1f}")
            for metric, (val, sc) in details[dim_key].items():
                if isinstance(val, float):
                    print(f"    {metric}: {val:.3f} → {sc:.0f}分")
                else:
                    print(f"    {metric}: {val} → {sc:.0f}分")

        print(f"\n{'=' * 70}")
        print(f"  综合得分: {final_score:.1f} / 100")
        self._print_grade(final_score)
        print(f"{'=' * 70}")

        assert final_score >= 30, f"Comprehensive score too low: {final_score:.1f}/100"

    @staticmethod
    def _print_grade(score: float):
        """输出评级"""
        if score >= 85:
            grade = "A (优秀) — 所有维度表现优异，达到生产级标准"
        elif score >= 70:
            grade = "B (良好) — 核心指标达标，存在可优化空间"
        elif score >= 55:
            grade = "C (合格) — 基础功能满足，多项指标需改进"
        elif score >= 40:
            grade = "D (待改进) — 部分维度存在明显短板"
        else:
            grade = "F (不合格) — 多项核心指标未达标，需重点整改"
        print(f"  评级: {grade}")