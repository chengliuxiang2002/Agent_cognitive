"""
认知记忆模块 - 长期记忆存储实现

基于 SQLite 的长期记忆持久化存储。
支持:
- 持久化存储用户偏好、行为模式
- 结构化查询
- 用户画像管理
- 交互记录追踪

性能优化 (PF-1 ~ PF-7):
- PF-1: 写入缓冲批量提交
- PF-2: SQLite 连接池复用
- PF-3: Redis 缓存层 (可选)
- PF-4: FTS5 全文索引
- PF-7: 数据库读写分离架构 (设计)

生产环境可替换为 PostgreSQL 实现。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

from .base import (
    BaseMemoryStore,
    BaseProfileStore,
    BaseInteractionStore,
    BasePatternStore,
)
from ..models.memory import (
    MemoryItem,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryType,
    MemoryImportance,
    UserProfile,
    InteractionRecord,
    BehaviorPattern,
    SceneContext,
)
from ..core.privacy import TieredEncryption

logger = logging.getLogger(__name__)

# PF-3: Redis 可选依赖
try:
    import redis.asyncio as aioredis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PF-1: 写入缓冲批量提交
# ═══════════════════════════════════════════════════════════════════════════════


class WriteBuffer:
    """异步写入缓冲器

    将逐条写入操作缓冲为批量 INSERT，减少数据库 IO 次数。
    触发条件:
    - 累积 N 条记录 (batch_size, 默认 50)
    - 距上次刷新超过 T 毫秒 (flush_interval_ms, 默认 100)
    """

    def __init__(
        self,
        flush_callback,
        batch_size: int = 50,
        flush_interval_ms: int = 100,
    ):
        self._buffer: deque = deque()
        self._flush_callback = flush_callback
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_ms / 1000.0
        self._lock = asyncio.Lock()
        self._last_flush = time.monotonic()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """启动后台定时刷新任务"""
        self._running = True
        self._flush_task = asyncio.create_task(self._timer_loop())

    async def stop(self):
        """停止后台刷新并强制清空缓冲区"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()

    async def add(self, item):
        """添加一条记录到缓冲区"""
        async with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self._batch_size:
                await self._flush_locked()

    async def _timer_loop(self):
        """定时刷新循环"""
        while self._running:
            await asyncio.sleep(self._flush_interval_s)
            async with self._lock:
                if self._buffer:
                    await self._flush_locked()

    async def _flush(self):
        """强制刷新缓冲区"""
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self):
        """在已持有锁的情况下执行刷新 (内部方法)"""
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()
        try:
            await self._flush_callback(batch)
            logger.debug(f"WriteBuffer flushed {len(batch)} records")
        except Exception as e:
            logger.error(f"WriteBuffer flush failed: {e}")
            # 失败时重新入队（简化处理：仅记录日志，不重试复杂逻辑）
            self._buffer.extendleft(reversed(batch))

    @property
    def pending_count(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# PF-2: SQLite 连接池
# ═══════════════════════════════════════════════════════════════════════════════


class ConnectionPool:
    """SQLite 连接池

    复用连接而非每次创建新连接，减少连接开销。
    在 WAL 模式下支持读写并发，通过连接复用提升吞吐量。
    """

    def __init__(self, db_path: str, pool_size: int = 3):
        self._db_path = db_path
        self._pool_size = pool_size
        self._pool: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(pool_size)

    def _create_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def acquire(self) -> sqlite3.Connection:
        """获取一个连接 (阻塞直到可用)"""
        self._semaphore.acquire()
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
                try:
                    conn.execute("SELECT 1")
                except sqlite3.Error:
                    conn = self._create_conn()
                return conn
            return self._create_conn()

    def release(self, conn: sqlite3.Connection):
        """归还连接到池中"""
        with self._lock:
            if len(self._pool) < self._pool_size:
                self._pool.append(conn)
            else:
                conn.close()
        self._semaphore.release()

    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._pool.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PF-3: Redis 缓存层
# ═══════════════════════════════════════════════════════════════════════════════


class RedisCache:
    """Redis 读缓存层

    针对长期记忆热点数据建立缓存，以 O(1) 时间复杂度返回。
    缓存键结构: mem:{user_id}:{memory_id}
    默认过期时间: 3600 秒 (1 小时)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Optional[aioredis.Redis] = None
        self._enabled = _REDIS_AVAILABLE

    async def connect(self):
        if not self._enabled:
            return
        try:
            self._client = aioredis.from_url(self._redis_url, decode_responses=False)
            await self._client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self._enabled = False

    async def disconnect(self):
        if self._client:
            await self._client.close()
            self._client = None

    def _cache_key(self, user_id: str, memory_id: str) -> str:
        return f"mem:{user_id}:{memory_id}"

    async def get(self, user_id: str, memory_id: str) -> Optional[dict]:
        if not self._enabled or not self._client:
            return None
        try:
            key = self._cache_key(user_id, memory_id)
            data = await self._client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
        return None

    async def set(self, user_id: str, memory_id: str, data: dict):
        if not self._enabled or not self._client:
            return
        try:
            key = self._cache_key(user_id, memory_id)
            await self._client.set(
                key, json.dumps(data, ensure_ascii=False), ex=self._ttl
            )
        except Exception as e:
            logger.debug(f"Redis set error: {e}")

    async def invalidate(self, user_id: str, memory_id: str):
        if not self._enabled or not self._client:
            return
        try:
            key = self._cache_key(user_id, memory_id)
            await self._client.delete(key)
        except Exception as e:
            logger.debug(f"Redis delete error: {e}")

    async def invalidate_user(self, user_id: str):
        if not self._enabled or not self._client:
            return
        try:
            pattern = f"mem:{user_id}:*"
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.debug(f"Redis scan delete error: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled


class LongTermMemoryStore(BaseMemoryStore):
    """长期记忆存储 - SQLite 实现 (含 PF-1~PF-4 性能优化 + SC-3 分级存储)

    PF-1: 写入缓冲批量提交 - WriteBuffer 异步批量写入
    PF-2: SQLite 连接池 - ConnectionPool 复用连接
    PF-3: Redis 缓存层 - RedisCache 热点数据缓存
    PF-4: FTS5 全文索引 - SQLite FTS5 加速关键词搜索
    SC-3: 数据分级存储 - CRITICAL/HIGH加密, TRANSIENT仅内存
    """

    # 批量写入 SQL 模板
    _BATCH_INSERT_SQL = """INSERT OR REPLACE INTO long_term_memories
        (id, user_id, memory_type, content, importance,
         created_at, last_accessed_at, last_updated_at,
         strength, decay_rate, min_strength, tags, context_keys,
         access_count, source, confidence, metadata, relevance_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    def __init__(
        self,
        db_path: str = "cognitive_memory.db",
        redis_url: str = "",
        write_buffer_size: int = 50,
        write_buffer_interval_ms: int = 100,
        pool_size: int = 3,
        encryption_key: Optional[bytes] = None,
    ):
        self._db_path = db_path

        # PF-2: 连接池
        self._pool = ConnectionPool(db_path, pool_size=pool_size)

        # PF-1: 写入缓冲
        self._write_buffer = WriteBuffer(
            flush_callback=self._batch_store,
            batch_size=write_buffer_size,
            flush_interval_ms=write_buffer_interval_ms,
        )

        # PF-3: Redis 缓存
        self._cache = RedisCache(redis_url) if redis_url else RedisCache()
        if not redis_url:
            self._cache._enabled = False  # 未配置 URL 时禁用

        # PF-4: FTS5 支持标记
        self._fts5_available = False

        # SC-3: 分级加密存储
        self._tiered_encryption = TieredEncryption(encryption_key=encryption_key)

        # SC-3: 仅内存存储的 TRANSIENT 数据缓存
        self._transient_store: dict[str, MemoryItem] = {}

        self._init_db()

    # ─── PF-2: 连接池辅助方法 ───────────────────────────

    def _acquire_conn(self) -> sqlite3.Connection:
        return self._pool.acquire()

    def _release_conn(self, conn: sqlite3.Connection):
        self._pool.release(conn)

    # ─── 数据库初始化 (含 PF-4 FTS5) ────────────────────

    def _init_db(self):
        """初始化数据库表结构 (含 FTS5 全文索引)"""
        conn = self._acquire_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '{}',
                    importance INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    last_updated_at TEXT NOT NULL,
                    strength REAL NOT NULL DEFAULT 1.0,
                    decay_rate REAL NOT NULL DEFAULT 0.01,
                    min_strength REAL NOT NULL DEFAULT 0.1,
                    tags TEXT NOT NULL DEFAULT '[]',
                    context_keys TEXT NOT NULL DEFAULT '[]',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    relevance_score REAL NOT NULL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_user_id
                    ON long_term_memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_ltm_memory_type
                    ON long_term_memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_ltm_strength
                    ON long_term_memories(strength);
                CREATE INDEX IF NOT EXISTS idx_ltm_created_at
                    ON long_term_memories(created_at);
                CREATE INDEX IF NOT EXISTS idx_ltm_relevance
                    ON long_term_memories(relevance_score);

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interaction_records (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL DEFAULT '',
                    intent TEXT NOT NULL DEFAULT '',
                    raw_input TEXT NOT NULL DEFAULT '',
                    processed_input TEXT NOT NULL DEFAULT '{}',
                    scene_context TEXT,
                    system_response TEXT NOT NULL DEFAULT '{}',
                    response_time_ms REAL NOT NULL DEFAULT 0.0,
                    user_satisfaction REAL,
                    was_successful INTEGER NOT NULL DEFAULT 1,
                    session_id TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_ir_user_id
                    ON interaction_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_ir_timestamp
                    ON interaction_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_ir_session_id
                    ON interaction_records(session_id);

                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL DEFAULT '',
                    trigger_conditions TEXT NOT NULL DEFAULT '{}',
                    expected_action TEXT NOT NULL DEFAULT '{}',
                    occurrence_count INTEGER NOT NULL DEFAULT 0,
                    success_rate REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    first_observed TEXT NOT NULL,
                    last_observed TEXT NOT NULL,
                    related_context_keys TEXT NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_bp_user_id
                    ON behavior_patterns(user_id);
                CREATE INDEX IF NOT EXISTS idx_bp_pattern_type
                    ON behavior_patterns(pattern_type);

                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL DEFAULT '',
                    prediction_data TEXT,
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_uf_user_id
                    ON user_feedback(user_id);
                CREATE INDEX IF NOT EXISTS idx_uf_prediction_id
                    ON user_feedback(prediction_id);
                CREATE INDEX IF NOT EXISTS idx_uf_created_at
                    ON user_feedback(created_at);
            """)

            # PF-4: 创建 FTS5 全文索引虚拟表
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(
                        id UNINDEXED,
                        user_id UNINDEXED,
                        content,
                        tags,
                        source,
                        tokenize='unicode61'
                    )
                """)
                self._fts5_available = True
                logger.info("FTS5 full-text search index initialized")
            except sqlite3.OperationalError as e:
                logger.warning(f"FTS5 not available, falling back to app-level search: {e}")
                self._fts5_available = False
        finally:
            self._release_conn(conn)

    async def start(self):
        """启动后台服务 (写入缓冲定时器, Redis 连接)"""
        await self._write_buffer.start()
        await self._cache.connect()

    async def stop(self):
        """停止后台服务并清理资源"""
        await self._write_buffer.stop()
        await self._cache.disconnect()
        self._pool.close_all()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        """将数据库行转换为 MemoryItem"""
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=json.loads(row["content"]),
            importance=MemoryImportance(row["importance"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
            strength=row["strength"],
            decay_rate=row["decay_rate"],
            min_strength=row["min_strength"],
            tags=json.loads(row["tags"]),
            context_keys=json.loads(row["context_keys"]),
            access_count=row["access_count"],
            source=row["source"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]),
        )

    # ─── PF-5: 预计算相关性分数 ─────────────────────────

    @staticmethod
    def _compute_relevance_score(item: MemoryItem) -> float:
        """预计算相关性分数 (写入时计算，查询时直接使用索引排序)

        算法: strength * 0.4 + importance_normalized * 0.3 + access_normalized * 0.3
        """
        importance_norm = item.importance.value / 5.0
        access_norm = min(item.access_count / 10.0, 1.0)
        return round(item.strength * 0.4 + importance_norm * 0.3 + access_norm * 0.3, 6)

    # ─── PF-1+PF-2: 优化的 store ────────────────────────

    async def store(self, item: MemoryItem) -> bool:
        """写入记忆 (通过 WriteBuffer 异步批量提交 + SC-3 分级存储)

        SC-3 分级策略:
        - TRANSIENT: 仅内存存储，不持久化
        - CRITICAL/HIGH: 端到端加密存储
        - MEDIUM/LOW: 明文持久化存储
        """
        # SC-3: TRANSIENT 数据仅保存在内存中
        if not self._tiered_encryption.should_persist(item.importance):
            self._transient_store[item.id] = item
            logger.debug(f"TRANSIENT memory stored in-memory: {item.id}")
            return True

        # SC-3: CRITICAL/HIGH 数据加密后存储
        if self._tiered_encryption.should_encrypt(item.importance):
            item = self._encrypt_item_content(item)

        # PF-5: 预计算相关性分数
        relevance = self._compute_relevance_score(item)
        await self._write_buffer.add((item, relevance))
        return True

    def _encrypt_item_content(self, item: MemoryItem) -> MemoryItem:
        """SC-3: 加密 MemoryItem 的 content 字段"""
        encrypted_content = self._tiered_encryption.encrypt_if_needed(
            item.content, item.importance
        )
        item.content = encrypted_content
        return item

    def _decrypt_item_content(self, item: MemoryItem) -> MemoryItem:
        """SC-3: 解密 MemoryItem 的 content 字段"""
        if isinstance(item.content, dict) and item.content.get("_encrypted", False):
            item.content = self._tiered_encryption.decrypt_if_needed(item.content)
        return item

    async def _batch_store(self, batch: list[tuple[MemoryItem, float]]):
        """批量写入回调 (由 WriteBuffer 触发)"""
        conn = self._acquire_conn()
        try:
            rows_data = []
            fts_data = []
            for item, relevance in batch:
                rows_data.append((
                    item.id, item.user_id, item.memory_type.value,
                    json.dumps(item.content, ensure_ascii=False),
                    item.importance.value,
                    item.created_at.isoformat(), item.last_accessed_at.isoformat(),
                    item.last_updated_at.isoformat(),
                    item.strength, item.decay_rate, item.min_strength,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.context_keys, ensure_ascii=False),
                    item.access_count, item.source, item.confidence,
                    json.dumps(item.metadata, ensure_ascii=False),
                    relevance,
                ))
                # PF-4: FTS5 索引数据
                fts_data.append((
                    item.id,
                    item.user_id,
                    json.dumps(item.content, ensure_ascii=False),
                    json.dumps(item.tags, ensure_ascii=False),
                    item.source,
                ))

            conn.executemany(self._BATCH_INSERT_SQL, rows_data)

            # PF-4: 同步更新 FTS5 索引
            if self._fts5_available:
                conn.executemany(
                    """INSERT OR REPLACE INTO memories_fts
                       (id, user_id, content, tags, source)
                       VALUES (?, ?, ?, ?, ?)""",
                    fts_data,
                )

            conn.commit()

            # PF-3: 更新 Redis 缓存
            for item, relevance in batch:
                await self._cache.set(item.user_id, item.id, self._item_to_cache_dict(item, relevance))

        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    def _item_to_cache_dict(self, item: MemoryItem, relevance: float) -> dict:
        """将 MemoryItem 序列化为缓存字典"""
        return {
            "id": item.id,
            "user_id": item.user_id,
            "memory_type": item.memory_type.value,
            "content": item.content,
            "importance": item.importance.value,
            "created_at": item.created_at.isoformat(),
            "last_accessed_at": item.last_accessed_at.isoformat(),
            "last_updated_at": item.last_updated_at.isoformat(),
            "strength": item.strength,
            "decay_rate": item.decay_rate,
            "min_strength": item.min_strength,
            "tags": item.tags,
            "context_keys": item.context_keys,
            "access_count": item.access_count,
            "source": item.source,
            "confidence": item.confidence,
            "metadata": item.metadata,
            "relevance_score": relevance,
        }

    def _cache_dict_to_item(self, data: dict) -> MemoryItem:
        """从缓存字典恢复 MemoryItem"""
        return MemoryItem(
            id=data["id"],
            user_id=data["user_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            importance=MemoryImportance(data["importance"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            last_updated_at=datetime.fromisoformat(data["last_updated_at"]),
            strength=data["strength"],
            decay_rate=data["decay_rate"],
            min_strength=data["min_strength"],
            tags=data["tags"],
            context_keys=data["context_keys"],
            access_count=data["access_count"],
            source=data["source"],
            confidence=data["confidence"],
            metadata=data["metadata"],
        )

    # ─── PF-2+PF-3: 优化的 retrieve ─────────────────────

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        # SC-3: 先检查 TRANSIENT 内存缓存
        if memory_id in self._transient_store:
            return self._transient_store[memory_id]

        # PF-1: 查询前先刷新写入缓冲
        await self._write_buffer._flush()

        conn = self._acquire_conn()
        try:
            row = conn.execute(
                "SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                item = self._row_to_item(row)
                # SC-3: 解密 CRITICAL/HIGH 数据
                item = self._decrypt_item_content(item)
                # 更新访问记录
                conn.execute(
                    """UPDATE long_term_memories
                       SET last_accessed_at = ?, access_count = access_count + 1
                       WHERE id = ?""",
                    (datetime.now().isoformat(), memory_id),
                )
                conn.commit()
                # 更新缓存
                relevance = self._compute_relevance_score(item)
                await self._cache.set(item.user_id, item.id, self._item_to_cache_dict(item, relevance))
                return item
        finally:
            self._release_conn(conn)
        return None

    # ─── PF-2+PF-4: 优化的 query ────────────────────────

    async def query(self, query: MemoryQuery) -> MemoryRetrievalResult:
        start_time = time.time()

        # PF-1: 查询前先刷新写入缓冲，确保数据一致性
        await self._write_buffer._flush()

        # PF-4: 关键词搜索优先使用 FTS5
        if query.keywords and self._fts5_available:
            result = await self._fts5_query(query, start_time)
        else:
            result = await self._standard_query(query, start_time)

        # SC-3: 解密所有返回的 CRITICAL/HIGH 数据
        result.items = [self._decrypt_item_content(item) for item in result.items]

        return result

    async def _standard_query(self, query: MemoryQuery, start_time: float) -> MemoryRetrievalResult:
        """标准 SQL 查询 (使用预计算 relevance_score 索引)"""
        conn = self._acquire_conn()
        try:
            sql = "SELECT * FROM long_term_memories WHERE 1=1"
            params: list = []

            if query.user_id:
                sql += " AND user_id = ?"
                params.append(query.user_id)

            if query.memory_types:
                placeholders = ",".join(["?" for _ in query.memory_types])
                sql += f" AND memory_type IN ({placeholders})"
                params.extend([mt.value for mt in query.memory_types])

            if query.min_strength > 0:
                sql += " AND strength >= ?"
                params.append(query.min_strength)

            if query.min_importance.value > 1:
                sql += " AND importance >= ?"
                params.append(query.min_importance.value)

            if query.time_range_days is not None:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(days=query.time_range_days)).isoformat()
                sql += " AND created_at >= ?"
                params.append(cutoff)

            # PF-5: 排序使用预计算 relevance_score 索引
            sort_map = {
                "recency": "last_accessed_at DESC",
                "strength": "strength DESC",
                "importance": "importance DESC",
                "relevance": "relevance_score DESC",
            }
            sql += f" ORDER BY {sort_map.get(query.sort_by, sort_map['relevance'])}"

            sql += " LIMIT ?"
            params.append(query.max_results)

            rows = conn.execute(sql, params).fetchall()
        finally:
            self._release_conn(conn)

        items = [self._row_to_item(row) for row in rows]

        # 应用层关键词过滤 (FTS5 不可用或未启用时的回退)
        if query.keywords and not self._fts5_available:
            items = [
                item for item in items
                if any(
                    kw.lower() in str(item.content).lower()
                    for kw in query.keywords
                )
            ]

        # 标签过滤
        if query.tags:
            items = [
                item for item in items
                if any(tag in item.tags for tag in query.tags)
            ]

        retrieval_time = (time.time() - start_time) * 1000

        return MemoryRetrievalResult(
            query=query,
            items=items,
            total_found=len(items),
            retrieval_time_ms=retrieval_time,
        )

    # ─── PF-4: FTS5 全文搜索 ────────────────────────────

    async def _fts5_query(self, query: MemoryQuery, start_time: float) -> MemoryRetrievalResult:
        """使用 FTS5 全文索引进行关键词搜索"""
        conn = self._acquire_conn()
        try:
            # 构建 FTS5 查询表达式
            fts_query = " OR ".join(f'"{kw}"' for kw in query.keywords)

            # 先通过 FTS5 搜索匹配的 memory_id
            fts_sql = """
                SELECT id FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            fts_rows = conn.execute(fts_sql, (fts_query, query.max_results * 2)).fetchall()
            matched_ids = [row["id"] for row in fts_rows]

            if not matched_ids:
                return MemoryRetrievalResult(
                    query=query,
                    items=[],
                    total_found=0,
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                )

            # 再用匹配的 ID 进行精确查询 + 排序
            placeholders = ",".join(["?" for _ in matched_ids])
            sql = f"SELECT * FROM long_term_memories WHERE id IN ({placeholders})"
            params = matched_ids

            if query.user_id:
                sql += " AND user_id = ?"
                params.append(query.user_id)

            if query.memory_types:
                type_placeholders = ",".join(["?" for _ in query.memory_types])
                sql += f" AND memory_type IN ({type_placeholders})"
                params.extend([mt.value for mt in query.memory_types])

            if query.min_strength > 0:
                sql += " AND strength >= ?"
                params.append(query.min_strength)

            # PF-5: 使用预计算 relevance_score 排序
            sql += " ORDER BY relevance_score DESC LIMIT ?"
            params.append(query.max_results)

            rows = conn.execute(sql, params).fetchall()
        finally:
            self._release_conn(conn)

        items = [self._row_to_item(row) for row in rows]

        if query.tags:
            items = [
                item for item in items
                if any(tag in item.tags for tag in query.tags)
            ]

        retrieval_time = (time.time() - start_time) * 1000

        return MemoryRetrievalResult(
            query=query,
            items=items,
            total_found=len(items),
            retrieval_time_ms=retrieval_time,
        )

    async def update(self, item: MemoryItem) -> bool:
        return await self.store(item)

    async def delete(self, memory_id: str) -> bool:
        conn = self._acquire_conn()
        try:
            # 先获取 user_id 用于缓存失效
            row = conn.execute(
                "SELECT user_id FROM long_term_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE id = ?", (memory_id,)
            )
            if cursor.rowcount > 0:
                # PF-4: 同步删除 FTS5 索引
                if self._fts5_available:
                    conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
                conn.commit()
                # PF-3: 缓存失效
                if row:
                    await self._cache.invalidate(row["user_id"], memory_id)
                return True
            return False
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def delete_by_user(self, user_id: str) -> int:
        # PF-1: 删除前先刷新缓冲，确保数据一致性
        await self._write_buffer._flush()

        conn = self._acquire_conn()
        try:
            # PF-4: 先获取要删除的 ID 列表
            rows = conn.execute(
                "SELECT id FROM long_term_memories WHERE user_id = ?", (user_id,)
            ).fetchall()
            ids = [row["id"] for row in rows]

            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE user_id = ?", (user_id,)
            )
            if ids and self._fts5_available:
                placeholders = ",".join(["?" for _ in ids])
                conn.execute(f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids)
            conn.commit()
            # PF-3: 批量缓存失效
            await self._cache.invalidate_user(user_id)
            return cursor.rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def get_user_memories(
        self, user_id: str, memory_type: Optional[str] = None, limit: int = 100
    ) -> list[MemoryItem]:
        # PF-1: 查询前先刷新写入缓冲
        await self._write_buffer._flush()

        sql = "SELECT * FROM long_term_memories WHERE user_id = ?"
        params: list = [user_id]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY last_accessed_at DESC LIMIT ?"
        params.append(limit)

        conn = self._acquire_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            self._release_conn(conn)

        return [self._row_to_item(row) for row in rows]

    async def cleanup_weak_memories(self, threshold: float = 0.1) -> int:
        conn = self._acquire_conn()
        try:
            # PF-4: 同步清理 FTS5 索引
            rows = conn.execute(
                "SELECT id FROM long_term_memories WHERE strength < ?", (threshold,)
            ).fetchall()
            ids = [row["id"] for row in rows]

            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE strength < ?", (threshold,)
            )
            if ids and self._fts5_available:
                placeholders = ",".join(["?" for _ in ids])
                conn.execute(f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids)
            conn.commit()
            return cursor.rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)


class ProfileStore(BaseProfileStore):
    """用户画像存储 - SQLite 实现 (PF-2: 连接池复用)"""

    def __init__(self, db_path: str = "cognitive_memory.db", pool: Optional[ConnectionPool] = None):
        self._db_path = db_path
        self._pool = pool or ConnectionPool(db_path, pool_size=2)
        self._ensure_table()

    def _acquire_conn(self) -> sqlite3.Connection:
        return self._pool.acquire()

    def _release_conn(self, conn: sqlite3.Connection):
        self._pool.release(conn)

    def _ensure_table(self):
        conn = self._acquire_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            self._release_conn(conn)

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        conn = self._acquire_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                data = json.loads(row["profile_data"])
                data["user_id"] = row["user_id"]
                data["created_at"] = datetime.fromisoformat(row["created_at"])
                data["updated_at"] = datetime.fromisoformat(row["updated_at"])
                return UserProfile(**data)
        finally:
            self._release_conn(conn)
        return None

    async def save_profile(self, profile: UserProfile) -> bool:
        conn = self._acquire_conn()
        try:
            profile_data = profile.to_dict()
            profile_data.pop("user_id", None)
            profile_data.pop("created_at", None)
            profile_data.pop("updated_at", None)

            conn.execute(
                """INSERT OR REPLACE INTO user_profiles
                   (user_id, profile_data, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    profile.user_id,
                    json.dumps(profile_data, ensure_ascii=False),
                    profile.created_at.isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def delete_profile(self, user_id: str) -> bool:
        conn = self._acquire_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)


class InteractionStore(BaseInteractionStore):
    """交互记录存储 - SQLite 实现 (PF-2: 连接池复用)"""

    def __init__(self, db_path: str = "cognitive_memory.db", pool: Optional[ConnectionPool] = None):
        self._db_path = db_path
        self._pool = pool or ConnectionPool(db_path, pool_size=2)
        self._ensure_table()

    def _acquire_conn(self) -> sqlite3.Connection:
        return self._pool.acquire()

    def _release_conn(self, conn: sqlite3.Connection):
        self._pool.release(conn)

    def _ensure_table(self):
        conn = self._acquire_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_records (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL DEFAULT '',
                    intent TEXT NOT NULL DEFAULT '',
                    raw_input TEXT NOT NULL DEFAULT '',
                    processed_input TEXT NOT NULL DEFAULT '{}',
                    scene_context TEXT,
                    system_response TEXT NOT NULL DEFAULT '{}',
                    response_time_ms REAL NOT NULL DEFAULT 0.0,
                    user_satisfaction REAL,
                    was_successful INTEGER NOT NULL DEFAULT 1,
                    session_id TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.commit()
        finally:
            self._release_conn(conn)

    async def record(self, interaction: InteractionRecord) -> bool:
        conn = self._acquire_conn()
        try:
            conn.execute(
                """INSERT INTO interaction_records
                   (id, user_id, timestamp, interaction_type, intent, raw_input,
                    processed_input, scene_context, system_response,
                    response_time_ms, user_satisfaction, was_successful,
                    session_id, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction.id,
                    interaction.user_id,
                    interaction.timestamp.isoformat(),
                    interaction.interaction_type,
                    interaction.intent,
                    interaction.raw_input,
                    json.dumps(interaction.processed_input, ensure_ascii=False),
                    json.dumps(interaction.scene_context.to_dict(), ensure_ascii=False)
                    if interaction.scene_context else None,
                    json.dumps(interaction.system_response, ensure_ascii=False),
                    interaction.response_time_ms,
                    interaction.user_satisfaction,
                    1 if interaction.was_successful else 0,
                    interaction.session_id,
                    json.dumps(interaction.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def get_recent(
        self, user_id: str, limit: int = 50
    ) -> list[InteractionRecord]:
        conn = self._acquire_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM interaction_records
                   WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        finally:
            self._release_conn(conn)

        return [self._row_to_record(row) for row in rows]

    async def get_by_session(
        self, session_id: str
    ) -> list[InteractionRecord]:
        conn = self._acquire_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM interaction_records
                   WHERE session_id = ?
                   ORDER BY timestamp ASC""",
                (session_id,),
            ).fetchall()
        finally:
            self._release_conn(conn)

        return [self._row_to_record(row) for row in rows]

    async def delete(self, record_id: str) -> bool:
        conn = self._acquire_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM interaction_records WHERE id = ?", (record_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    def _row_to_record(self, row: sqlite3.Row) -> InteractionRecord:
        scene_ctx = None
        if row["scene_context"]:
            ctx_data = json.loads(row["scene_context"])
            scene_ctx = SceneContext(**ctx_data)

        return InteractionRecord(
            id=row["id"],
            user_id=row["user_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            interaction_type=row["interaction_type"],
            intent=row["intent"],
            raw_input=row["raw_input"],
            processed_input=json.loads(row["processed_input"]),
            scene_context=scene_ctx,
            system_response=json.loads(row["system_response"]),
            response_time_ms=row["response_time_ms"],
            user_satisfaction=row["user_satisfaction"],
            was_successful=bool(row["was_successful"]),
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]),
        )


class PatternStore(BasePatternStore):
    """行为模式存储 - SQLite 实现 (PF-2: 连接池复用)"""

    def __init__(self, db_path: str = "cognitive_memory.db", pool: Optional[ConnectionPool] = None):
        self._db_path = db_path
        self._pool = pool or ConnectionPool(db_path, pool_size=2)
        self._ensure_table()

    def _acquire_conn(self) -> sqlite3.Connection:
        return self._pool.acquire()

    def _release_conn(self, conn: sqlite3.Connection):
        self._pool.release(conn)

    def _ensure_table(self):
        conn = self._acquire_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL DEFAULT '',
                    trigger_conditions TEXT NOT NULL DEFAULT '{}',
                    expected_action TEXT NOT NULL DEFAULT '{}',
                    occurrence_count INTEGER NOT NULL DEFAULT 0,
                    success_rate REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    first_observed TEXT NOT NULL,
                    last_observed TEXT NOT NULL,
                    related_context_keys TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.commit()
        finally:
            self._release_conn(conn)

    async def save_pattern(self, pattern: BehaviorPattern) -> bool:
        conn = self._acquire_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO behavior_patterns
                   (id, user_id, pattern_name, pattern_type,
                    trigger_conditions, expected_action,
                    occurrence_count, success_rate, confidence,
                    first_observed, last_observed, related_context_keys)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern.id,
                    pattern.user_id,
                    pattern.pattern_name,
                    pattern.pattern_type,
                    json.dumps(pattern.trigger_conditions, ensure_ascii=False),
                    json.dumps(pattern.expected_action, ensure_ascii=False),
                    pattern.occurrence_count,
                    pattern.success_rate,
                    pattern.confidence,
                    pattern.first_observed.isoformat(),
                    pattern.last_observed.isoformat(),
                    json.dumps(pattern.related_context_keys, ensure_ascii=False),
                ),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def get_patterns(
        self, user_id: str, pattern_type: Optional[str] = None
    ) -> list[BehaviorPattern]:
        sql = "SELECT * FROM behavior_patterns WHERE user_id = ?"
        params: list = [user_id]
        if pattern_type:
            sql += " AND pattern_type = ?"
            params.append(pattern_type)
        sql += " ORDER BY confidence DESC"

        conn = self._acquire_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            self._release_conn(conn)

        return [self._row_to_pattern(row) for row in rows]

    async def get_all_patterns(
        self, limit: int = 200
    ) -> list[BehaviorPattern]:
        """获取所有用户的行为模式（用于管理界面图谱展示）"""
        sql = "SELECT * FROM behavior_patterns ORDER BY confidence DESC LIMIT ?"
        conn = self._acquire_conn()
        try:
            rows = conn.execute(sql, (limit,)).fetchall()
        finally:
            self._release_conn(conn)
        return [self._row_to_pattern(row) for row in rows]

    async def get_pattern_by_name(
        self, user_id: str, pattern_name: str
    ) -> Optional[BehaviorPattern]:
        conn = self._acquire_conn()
        try:
            row = conn.execute(
                """SELECT * FROM behavior_patterns
                   WHERE user_id = ? AND pattern_name = ?""",
                (user_id, pattern_name),
            ).fetchone()
            if row:
                return self._row_to_pattern(row)
        finally:
            self._release_conn(conn)
        return None

    async def delete_pattern(self, pattern_id: str) -> bool:
        conn = self._acquire_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM behavior_patterns WHERE id = ?", (pattern_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    def _row_to_pattern(self, row: sqlite3.Row) -> BehaviorPattern:
        return BehaviorPattern(
            id=row["id"],
            user_id=row["user_id"],
            pattern_name=row["pattern_name"],
            pattern_type=row["pattern_type"],
            trigger_conditions=json.loads(row["trigger_conditions"]),
            expected_action=json.loads(row["expected_action"]),
            occurrence_count=row["occurrence_count"],
            success_rate=row["success_rate"],
            confidence=row["confidence"],
            first_observed=datetime.fromisoformat(row["first_observed"]),
            last_observed=datetime.fromisoformat(row["last_observed"]),
            related_context_keys=json.loads(row["related_context_keys"]),
        )


class FeedbackStore:
    """用户反馈存储 - SQLite 实现 (PF-2: 连接池复用)

    用于存储用户对预测结果的二元反馈（点赞/踩），
    反馈数据用于动态调整模式识别的置信度权重算法。
    """

    def __init__(self, db_path: str = "cognitive_memory.db", pool: Optional[ConnectionPool] = None):
        self._db_path = db_path
        self._pool = pool or ConnectionPool(db_path, pool_size=2)
        self._ensure_table()

    def _acquire_conn(self) -> sqlite3.Connection:
        return self._pool.acquire()

    def _release_conn(self, conn: sqlite3.Connection):
        self._pool.release(conn)

    def _ensure_table(self):
        conn = self._acquire_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL DEFAULT '',
                    prediction_data TEXT,
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            self._release_conn(conn)

    async def record_feedback(
        self,
        feedback_id: str,
        user_id: str,
        prediction_id: str,
        feedback_type: str,
        prediction_data: Optional[dict] = None,
        comment: str = "",
    ) -> bool:
        conn = self._acquire_conn()
        try:
            conn.execute(
                """INSERT INTO user_feedback
                   (id, user_id, prediction_id, feedback_type,
                    prediction_data, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    user_id,
                    prediction_id,
                    feedback_type,
                    json.dumps(prediction_data, ensure_ascii=False) if prediction_data else None,
                    comment,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_conn(conn)

    async def get_feedback_stats(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> dict:
        """获取反馈统计"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = self._acquire_conn()
        try:
            if user_id:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes,
                        SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as dislikes
                       FROM user_feedback
                       WHERE user_id = ? AND created_at >= ?""",
                    (user_id, cutoff),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes,
                        SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as dislikes
                       FROM user_feedback
                       WHERE created_at >= ?""",
                    (cutoff,),
                ).fetchone()
        finally:
            self._release_conn(conn)

        total = row["total"] or 0
        likes = row["likes"] or 0
        dislikes = row["dislikes"] or 0

        return {
            "total": total,
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": round(likes / total, 4) if total > 0 else 0.0,
            "period_days": days,
        }

    async def get_user_feedback(
        self, user_id: str, limit: int = 50
    ) -> list[dict]:
        conn = self._acquire_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM user_feedback
                   WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        finally:
            self._release_conn(conn)

        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "prediction_id": row["prediction_id"],
                "feedback_type": row["feedback_type"],
                "prediction_data": json.loads(row["prediction_data"]) if row["prediction_data"] else None,
                "comment": row["comment"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# PF-7: 数据库读写分离架构设计
# ═══════════════════════════════════════════════════════════════════════════════
#
# 目标: 在生产环境引入 PostgreSQL 主从架构，实现数据库读写分离。
#
# 架构设计:
# ┌─────────────────────────────────────────────────────────────┐
# │                     MemoryManager                           │
# │  ┌──────────────┐                    ┌──────────────────┐   │
# │  │ 写操作路由    │                    │ 读操作路由        │   │
# │  │ store()      │                    │ query()          │   │
# │  │ update()     │                    │ retrieve()       │   │
# │  │ delete()     │                    │ get_user_*()     │   │
# │  └──────┬───────┘                    └────────┬─────────┘   │
# └─────────┼──────────────────────────────────────┼────────────┘
#           │                                      │
#           ▼                                      ▼
# ┌─────────────────┐                   ┌──────────────────────┐
# │  PostgreSQL     │  ─── WAL 复制 ──▶ │  PostgreSQL 只读副本  │
# │  主库 (Master)  │                   │  (Read Replica 1..N) │
# │  - 写入         │                   │  - 查询              │
# │  - DDL          │                   │  - 统计              │
# └─────────────────┘                   └──────────────────────┘
#
# 实施策略:
# 1. 连接抽象层:
#    - 创建 ReadWriteRouter 类，封装主库和只读副本的连接管理
#    - 主库连接: 用于 INSERT / UPDATE / DELETE / DDL
#    - 副本连接池: 用于 SELECT 查询，支持负载均衡 (round-robin)
#
# 2. 路由规则:
#    - 写操作 → 主库: store(), update(), delete(), cleanup_*()
#    - 读操作 → 副本: query(), retrieve(), get_user_memories()
#    - 事务操作 → 主库: 包含写操作的事务必须在主库执行
#
# 3. 连接配置:
#    class ReadWriteRouter:
#        def __init__(self,
#                     master_url: str,       # postgresql://user:pass@host:5432/db
#                     replica_urls: list[str], # [postgresql://user:pass@host2:5432/db, ...]
#                     pool_size: int = 5):
#            self._master_pool = asyncpg.create_pool(master_url, min_size=2, max_size=pool_size)
#            self._replica_pools = [asyncpg.create_pool(url, ...) for url in replica_urls]
#            self._replica_index = 0  # round-robin
#
# 4. 负载均衡:
#    - Round-robin 轮询分发读请求到多个只读副本
#    - 副本健康检查: 定期 ping 检测可用性，自动剔除不可用副本
#    - 读延迟容忍: 接受主从复制延迟 (通常 < 100ms)
#
# 5. 降级策略:
#    - 所有副本不可用时: 自动降级到主库查询
#    - 主库不可用时: 写操作失败，读操作继续使用副本 (降级模式)
#
# 6. 迁移计划:
#    Phase 1: 实现 ReadWriteRouter 抽象层 (当前 SQLite 兼容)
#    Phase 2: 迁移到 PostgreSQL 单实例 (验证功能)
#    Phase 3: 添加只读副本，启用读写分离
#    Phase 4: 性能调优，监控告警
#
# 预期性能提升:
# - 查询吞吐量: 线性扩展 (与副本数成正比)
# - 写入性能: 不受读负载影响
# - 系统整体吞吐量: 从 87 ops/s → 500+ ops/s