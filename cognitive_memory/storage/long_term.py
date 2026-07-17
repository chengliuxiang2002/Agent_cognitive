"""
认知记忆模块 - 长期记忆存储实现

基于 SQLite 的长期记忆持久化存储。
支持:
- 持久化存储用户偏好、行为模式
- 结构化查询
- 用户画像管理
- 交互记录追踪

生产环境可替换为 PostgreSQL 实现。
"""

from __future__ import annotations

import json
import sqlite3
import time
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


class LongTermMemoryStore(BaseMemoryStore):
    """长期记忆存储 - SQLite 实现"""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
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
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_user_id
                    ON long_term_memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_ltm_memory_type
                    ON long_term_memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_ltm_strength
                    ON long_term_memories(strength);
                CREATE INDEX IF NOT EXISTS idx_ltm_created_at
                    ON long_term_memories(created_at);

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

    async def store(self, item: MemoryItem) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO long_term_memories
                   (id, user_id, memory_type, content, importance,
                    created_at, last_accessed_at, last_updated_at,
                    strength, decay_rate, min_strength, tags, context_keys,
                    access_count, source, confidence, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
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
                ),
            )
        return True

    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                item = self._row_to_item(row)
                # 更新访问记录
                conn.execute(
                    """UPDATE long_term_memories
                       SET last_accessed_at = ?, access_count = access_count + 1
                       WHERE id = ?""",
                    (datetime.now().isoformat(), memory_id),
                )
                return item
        return None

    async def query(self, query: MemoryQuery) -> MemoryRetrievalResult:
        start_time = time.time()

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

        # 排序
        sort_map = {
            "recency": "last_accessed_at DESC",
            "strength": "strength DESC",
            "importance": "importance DESC",
            "relevance": "(strength * 0.4 + importance * 1.0 / 5 * 0.3 + MIN(access_count, 10) * 1.0 / 10 * 0.3) DESC",
        }
        sql += f" ORDER BY {sort_map.get(query.sort_by, sort_map['relevance'])}"

        sql += f" LIMIT ?"
        params.append(query.max_results)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        items = [self._row_to_item(row) for row in rows]

        # 关键词过滤（SQLite不支持复杂文本搜索，在应用层处理）
        if query.keywords:
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

    async def update(self, item: MemoryItem) -> bool:
        return await self.store(item)

    async def delete(self, memory_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    async def delete_by_user(self, user_id: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE user_id = ?", (user_id,)
            )
            return cursor.rowcount

    async def get_user_memories(
        self, user_id: str, memory_type: Optional[str] = None, limit: int = 100
    ) -> list[MemoryItem]:
        sql = "SELECT * FROM long_term_memories WHERE user_id = ?"
        params: list = [user_id]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY last_accessed_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_item(row) for row in rows]

    async def cleanup_weak_memories(self, threshold: float = 0.1) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM long_term_memories WHERE strength < ?", (threshold,)
            )
            return cursor.rowcount


class ProfileStore(BaseProfileStore):
    """用户画像存储 - SQLite 实现"""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                data = json.loads(row["profile_data"])
                data["user_id"] = row["user_id"]
                data["created_at"] = datetime.fromisoformat(row["created_at"])
                data["updated_at"] = datetime.fromisoformat(row["updated_at"])
                return UserProfile(**data)
        return None

    async def save_profile(self, profile: UserProfile) -> bool:
        with self._get_conn() as conn:
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
        return True

    async def delete_profile(self, user_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            return cursor.rowcount > 0


class InteractionStore(BaseInteractionStore):
    """交互记录存储 - SQLite 实现"""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
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

    async def record(self, interaction: InteractionRecord) -> bool:
        with self._get_conn() as conn:
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
        return True

    async def get_recent(
        self, user_id: str, limit: int = 50
    ) -> list[InteractionRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM interaction_records
                   WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    async def get_by_session(
        self, session_id: str
    ) -> list[InteractionRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM interaction_records
                   WHERE session_id = ?
                   ORDER BY timestamp ASC""",
                (session_id,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    async def delete(self, record_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM interaction_records WHERE id = ?", (record_id,)
            )
            return cursor.rowcount > 0

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
    """行为模式存储 - SQLite 实现"""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
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

    async def save_pattern(self, pattern: BehaviorPattern) -> bool:
        with self._get_conn() as conn:
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
        return True

    async def get_patterns(
        self, user_id: str, pattern_type: Optional[str] = None
    ) -> list[BehaviorPattern]:
        sql = "SELECT * FROM behavior_patterns WHERE user_id = ?"
        params: list = [user_id]
        if pattern_type:
            sql += " AND pattern_type = ?"
            params.append(pattern_type)
        sql += " ORDER BY confidence DESC"

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def get_all_patterns(
        self, limit: int = 200
    ) -> list[BehaviorPattern]:
        """获取所有用户的行为模式（用于管理界面图谱展示）"""
        sql = "SELECT * FROM behavior_patterns ORDER BY confidence DESC LIMIT ?"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [self._row_to_pattern(row) for row in rows]

    async def get_pattern_by_name(
        self, user_id: str, pattern_name: str
    ) -> Optional[BehaviorPattern]:
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT * FROM behavior_patterns
                   WHERE user_id = ? AND pattern_name = ?""",
                (user_id, pattern_name),
            ).fetchone()
            if row:
                return self._row_to_pattern(row)
        return None

    async def delete_pattern(self, pattern_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM behavior_patterns WHERE id = ?", (pattern_id,)
            )
            return cursor.rowcount > 0

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
    """用户反馈存储 - SQLite 实现

    用于存储用户对预测结果的二元反馈（点赞/踩），
    反馈数据用于动态调整模式识别的置信度权重算法。
    """

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
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

    async def record_feedback(
        self,
        feedback_id: str,
        user_id: str,
        prediction_id: str,
        feedback_type: str,
        prediction_data: Optional[dict] = None,
        comment: str = "",
    ) -> bool:
        with self._get_conn() as conn:
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
        return True

    async def get_feedback_stats(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> dict:
        """获取反馈统计"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM user_feedback
                   WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()

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