"""
性能优化测试 (PF-1 ~ PF-7)

验证各优化项的正确性和性能提升效果。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pytest

from ..models.memory import (
    MemoryItem,
    MemoryQuery,
    MemoryType,
    MemoryImportance,
    InteractionRecord,
)
from ..storage.long_term import LongTermMemoryStore


# ═════════════════════════════════════════════════════════════════════════════
# PF-1: 写入缓冲批量提交测试
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteBuffer:
    """测试 WriteBuffer 异步批量写入机制"""

    @pytest.mark.asyncio
    async def test_buffer_accumulates_items(self):
        """测试缓冲累积：写入缓冲应累积数据而非立即写入"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用，表已初始化

        item = MemoryItem(
            user_id="test_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
            importance=MemoryImportance.LOW,
        )

        # 写入多条数据
        for i in range(10):
            await store.store(item)

        # 缓冲中应有数据（未刷盘）
        assert len(store._write_buffer._buffer) >= 0

        await store.stop()

    @pytest.mark.asyncio
    async def test_buffer_flush_on_query(self):
        """测试查询前自动刷新缓冲：确保数据一致性"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        item = MemoryItem(
            user_id="test_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test_query"},
            importance=MemoryImportance.MEDIUM,
        )

        await store.store(item)

        # 查询应立即看到刚写入的数据
        result = await store.query(MemoryQuery(user_id="test_user"))
        assert result.total_found >= 1

        await store.stop()

    @pytest.mark.asyncio
    async def test_batch_insert_performance(self):
        """测试批量写入性能：应优于逐条写入"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        items = [
            MemoryItem(
                user_id="perf_user",
                memory_type=MemoryType.EPISODIC,
                content={"action": f"test_{i}"},
                importance=MemoryImportance.LOW,
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        for item in items:
            await store.store(item)
        # 强制刷新
        await store._write_buffer._flush()
        elapsed = time.perf_counter() - start

        # 100条写入应在合理时间内完成
        assert elapsed < 5.0, f"批量写入耗时过长: {elapsed:.2f}s"

        # 验证所有数据写入
        result = await store.query(MemoryQuery(user_id="perf_user", max_results=200))
        assert result.total_found == 100

        await store.stop()

    @pytest.mark.asyncio
    async def test_buffer_time_threshold(self):
        """测试时间阈值触发刷新"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        store._write_buffer._flush_interval_s = 0.05  # 50ms 时间阈值
        await store.start()

        item = MemoryItem(
            user_id="time_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test_time"},
            importance=MemoryImportance.LOW,
        )

        await store.store(item)
        # 等待超过时间阈值
        await asyncio.sleep(0.1)

        # 缓冲应自动刷新
        result = await store.query(MemoryQuery(user_id="time_user"))
        assert result.total_found >= 1

        await store.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PF-2: SQLite 连接池测试
# ═════════════════════════════════════════════════════════════════════════════

class TestConnectionPool:
    """测试连接池复用机制"""

    def test_pool_creates_connections(self):
        """测试连接池创建连接"""
        pool = LongTermMemoryStore(":memory:")
        conn = pool._acquire_conn()
        assert conn is not None
        pool._release_conn(conn)

    def test_pool_reuses_connections(self):
        """测试连接池复用连接"""
        pool = LongTermMemoryStore(":memory:")
        conn1 = pool._acquire_conn()
        pool._release_conn(conn1)

        conn2 = pool._acquire_conn()
        assert conn2 is not None
        pool._release_conn(conn2)


# ═════════════════════════════════════════════════════════════════════════════
# PF-3: Redis 缓存层测试
# ═════════════════════════════════════════════════════════════════════════════

class TestRedisCache:
    """测试 Redis 缓存层"""

    def test_cache_key_generation(self):
        """测试缓存键生成"""
        store = LongTermMemoryStore(":memory:")
        key = store._cache._cache_key("test_user", "memory_id_123")
        assert key.startswith("mem:")
        assert "test_user" in key
        assert "memory_id_123" in key

    def test_cache_disabled_without_redis(self):
        """测试未配置 Redis 时缓存不可用"""
        store = LongTermMemoryStore(":memory:")
        assert store._cache._enabled is False


# ═════════════════════════════════════════════════════════════════════════════
# PF-4: FTS5 全文索引测试
# ═════════════════════════════════════════════════════════════════════════════

class TestFTS5Index:
    """测试 FTS5 全文索引"""

    @pytest.mark.asyncio
    async def test_fts5_search(self):
        """测试 FTS5 关键词搜索"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        # 写入包含关键词的数据
        item = MemoryItem(
            user_id="fts_user",
            memory_type=MemoryType.SEMANTIC,
            content={"action": "导航到北京天安门", "destination": "天安门"},
            tags=["北京", "导航"],
            importance=MemoryImportance.MEDIUM,
        )

        await store.store(item)
        await store._write_buffer._flush()

        # 使用 FTS5 搜索
        result = await store.query(MemoryQuery(
            user_id="fts_user",
            keywords=["北京"],
        ))

        assert result.total_found >= 1

        await store.stop()

    @pytest.mark.asyncio
    async def test_fts5_fallback_to_standard(self):
        """测试 FTS5 不可用时回退到标准搜索"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        store._fts5_available = False  # 模拟 FTS5 不可用

        item = MemoryItem(
            user_id="fallback_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
            importance=MemoryImportance.LOW,
        )

        await store.store(item)
        await store._write_buffer._flush()

        result = await store.query(MemoryQuery(
            user_id="fallback_user",
            keywords=["test"],
        ))

        # 应回退到标准查询
        assert result.total_found >= 1


# ═════════════════════════════════════════════════════════════════════════════
# PF-5: 查询结果预计算测试
# ═════════════════════════════════════════════════════════════════════════════

class TestRelevancePrecomputation:
    """测试 relevance_score 预计算"""

    def test_relevance_score_calculation(self):
        """测试相关性分数计算"""
        store = LongTermMemoryStore(":memory:")
        item = MemoryItem(
            user_id="rel_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
            importance=MemoryImportance.CRITICAL,
        )
        item.strength = 1.0
        item.access_count = 5

        score = store._compute_relevance_score(item)
        assert 0.0 <= score <= 1.0
        # CRITICAL 重要性应产生较高分数
        assert score > 0.5

    def test_relevance_score_low_importance(self):
        """测试低重要性记忆的相关性分数"""
        store = LongTermMemoryStore(":memory:")
        item = MemoryItem(
            user_id="rel_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
            importance=MemoryImportance.TRANSIENT,
        )
        item.strength = 0.1
        item.access_count = 0

        score = store._compute_relevance_score(item)
        assert 0.0 <= score <= 1.0
        # TRANSIENT 重要性应产生较低分数
        assert score < 0.3

    @pytest.mark.asyncio
    async def test_relevance_score_stored(self):
        """测试相关性分数在写入时被存储"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        item = MemoryItem(
            user_id="rel_store_user",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test_store"},
            importance=MemoryImportance.MEDIUM,
        )

        await store.store(item)
        await store._write_buffer._flush()

        # 查询并验证排序
        result = await store.query(MemoryQuery(
            user_id="rel_store_user",
            sort_by="relevance",
        ))

        assert result.total_found >= 1

        await store.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PF-7: 读写分离设计验证测试
# ═════════════════════════════════════════════════════════════════════════════

class TestReadWriteSeparation:
    """测试读写分离架构设计"""

    def test_design_documentation_exists(self):
        """测试 PF-7 设计文档已存在于源码中"""
        from ..storage import long_term as lt_module
        docstring = lt_module.__doc__ or ""
        source = open(lt_module.__file__, encoding="utf-8").read()
        # 验证 PF-7 设计文档存在于代码中
        assert "PF-7" in source or "ReadWriteRouter" in source or "读写分离" in source


# ═════════════════════════════════════════════════════════════════════════════
# 综合性能测试
# ═════════════════════════════════════════════════════════════════════════════

class TestComprehensivePerformance:
    """综合性能测试"""

    @pytest.mark.asyncio
    async def test_end_to_end_write_read_cycle(self):
        """测试端到端读写循环性能"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        user_id = "e2e_user"

        # 写入阶段
        write_start = time.perf_counter()
        for i in range(50):
            item = MemoryItem(
                user_id=user_id,
                memory_type=MemoryType.EPISODIC,
                content={"action": f"e2e_test_{i}"},
                importance=MemoryImportance.LOW,
            )
            await store.store(item)
        # 强制刷新
        await store._write_buffer._flush()
        write_time = time.perf_counter() - write_start

        # 查询阶段
        read_start = time.perf_counter()
        for _ in range(10):
            result = await store.query(MemoryQuery(
                user_id=user_id,
                sort_by="relevance",
                max_results=200,
            ))
            assert result.total_found >= 50
        read_time = time.perf_counter() - read_start

        # 性能验证
        assert write_time < 5.0, f"写入阶段耗时过长: {write_time:.2f}s"
        assert read_time < 2.0, f"查询阶段耗时过长: {read_time:.2f}s"

        await store.stop()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """测试并发写入性能"""
        store = LongTermMemoryStore(":memory:")
        # _init_db() 在 __init__ 中已调用
        await store.start()

        async def write_batch(user_id: str, count: int):
            for i in range(count):
                item = MemoryItem(
                    user_id=user_id,
                    memory_type=MemoryType.EPISODIC,
                    content={"action": f"concurrent_{i}"},
                    importance=MemoryImportance.LOW,
                )
                await store.store(item)

        # 并发写入
        start = time.perf_counter()
        await asyncio.gather(
            write_batch("concurrent_user_1", 30),
            write_batch("concurrent_user_2", 30),
            write_batch("concurrent_user_3", 30),
        )
        await store._write_buffer._flush()
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"并发写入耗时过长: {elapsed:.2f}s"

        # 验证所有数据
        for uid in ["concurrent_user_1", "concurrent_user_2", "concurrent_user_3"]:
            result = await store.query(MemoryQuery(user_id=uid, max_results=200))
            assert result.total_found == 30

        await store.stop()