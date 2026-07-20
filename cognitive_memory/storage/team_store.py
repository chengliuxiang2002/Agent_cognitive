"""
认知记忆模块 - 团队记忆存储实现

基于 SQLite 的团队记忆持久化存储。
支持:
- 团队CRUD操作
- 团队记忆CRUD操作
- 权限校验
- 个人记忆与团队记忆隔离
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from typing import Any, Optional

from ..models.team_memory import (
    Team,
    TeamMember,
    TeamMemory,
    TeamMemoryType,
    TeamPermission,
    TeamMemoryQuery,
)


class TeamStore:
    """团队存储 - SQLite 实现"""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    department TEXT NOT NULL DEFAULT '',
                    members TEXT NOT NULL DEFAULT '[]',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_teams_name
                    ON teams(name);
                CREATE INDEX IF NOT EXISTS idx_teams_department
                    ON teams(department);

                CREATE TABLE IF NOT EXISTS team_memories (
                    id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    memory_type TEXT NOT NULL DEFAULT 'general',
                    content TEXT NOT NULL DEFAULT '{}',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    keywords TEXT NOT NULL DEFAULT '[]',
                    importance INTEGER NOT NULL DEFAULT 3,
                    is_public INTEGER NOT NULL DEFAULT 1,
                    allowed_members TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_tm_team_id
                    ON team_memories(team_id);
                CREATE INDEX IF NOT EXISTS idx_tm_memory_type
                    ON team_memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_tm_updated_at
                    ON team_memories(updated_at);
            """)

    # ─── 团队管理 ────────────────────────────────────────

    async def create_team(self, team: Team) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO teams
                   (id, name, description, department, members, created_by,
                    created_at, updated_at, tags, is_active, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    team.id,
                    team.name,
                    team.description,
                    team.department,
                    json.dumps([m.to_dict() for m in team.members], ensure_ascii=False),
                    team.created_by,
                    team.created_at.isoformat(),
                    team.updated_at.isoformat(),
                    json.dumps(team.tags, ensure_ascii=False),
                    1 if team.is_active else 0,
                    json.dumps(team.metadata, ensure_ascii=False),
                ),
            )
        return True

    async def get_team(self, team_id: str) -> Optional[Team]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM teams WHERE id = ?", (team_id,)
            ).fetchone()
            if row:
                return self._row_to_team(row)
        return None

    async def get_user_teams(self, user_id: str) -> list[Team]:
        """获取用户所属的所有团队"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM teams WHERE is_active = 1"
            ).fetchall()

        teams = []
        for row in rows:
            members = json.loads(row["members"])
            if any(m["user_id"] == user_id for m in members):
                teams.append(self._row_to_team(row))
        return teams

    async def update_team(self, team: Team) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE teams SET
                   name=?, description=?, department=?, members=?,
                   updated_at=?, tags=?, is_active=?, metadata=?
                   WHERE id=?""",
                (
                    team.name,
                    team.description,
                    team.department,
                    json.dumps([m.to_dict() for m in team.members], ensure_ascii=False),
                    datetime.now().isoformat(),
                    json.dumps(team.tags, ensure_ascii=False),
                    1 if team.is_active else 0,
                    json.dumps(team.metadata, ensure_ascii=False),
                    team.id,
                ),
            )
        return True

    async def delete_team(self, team_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM teams WHERE id = ?", (team_id,)
            )
            if cursor.rowcount > 0:
                conn.execute(
                    "DELETE FROM team_memories WHERE team_id = ?", (team_id,)
                )
                return True
        return False

    async def add_member(
        self, team_id: str, member: TeamMember
    ) -> bool:
        team = await self.get_team(team_id)
        if team is None:
            return False
        if team.has_member(member.user_id):
            return False
        team.members.append(member)
        return await self.update_team(team)

    async def remove_member(self, team_id: str, user_id: str) -> bool:
        team = await self.get_team(team_id)
        if team is None:
            return False
        team.members = [m for m in team.members if m.user_id != user_id]
        return await self.update_team(team)

    async def update_member_permission(
        self, team_id: str, user_id: str, permission: TeamPermission
    ) -> bool:
        team = await self.get_team(team_id)
        if team is None:
            return False
        for m in team.members:
            if m.user_id == user_id:
                m.permission = permission
                return await self.update_team(team)
        return False

    # ─── 团队记忆管理 ────────────────────────────────────

    async def store_memory(self, memory: TeamMemory) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO team_memories
                   (id, team_id, title, memory_type, content,
                    created_by, created_at, updated_by, updated_at,
                    tags, keywords, importance, is_public, allowed_members,
                    version, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.team_id,
                    memory.title,
                    memory.memory_type.value,
                    json.dumps(memory.content, ensure_ascii=False),
                    memory.created_by,
                    memory.created_at.isoformat(),
                    memory.updated_by,
                    memory.updated_at.isoformat(),
                    json.dumps(memory.tags, ensure_ascii=False),
                    json.dumps(memory.keywords, ensure_ascii=False),
                    memory.importance,
                    1 if memory.is_public else 0,
                    json.dumps(memory.allowed_members, ensure_ascii=False),
                    memory.version,
                    json.dumps(memory.metadata, ensure_ascii=False),
                ),
            )
        return True

    async def query_memories(
        self, query: TeamMemoryQuery
    ) -> dict[str, Any]:
        start_time = time.time()

        sql = "SELECT * FROM team_memories WHERE team_id = ?"
        params: list = [query.team_id]

        if query.memory_types:
            placeholders = ",".join(["?" for _ in query.memory_types])
            sql += f" AND memory_type IN ({placeholders})"
            params.extend([mt.value for mt in query.memory_types])

        if query.tags:
            # 在应用层过滤标签
            pass

        if query.time_range_days is not None:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=query.time_range_days)).isoformat()
            sql += " AND updated_at >= ?"
            params.append(cutoff)

        sort_map = {
            "updated_at": "updated_at DESC",
            "created_at": "created_at DESC",
            "importance": "importance DESC",
        }
        sql += f" ORDER BY {sort_map.get(query.sort_by, sort_map['updated_at'])}"
        sql += " LIMIT ?"
        params.append(query.max_results)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        items = [self._row_to_memory(row) for row in rows]

        # 权限过滤：检查用户是否有权查看每条记忆
        team = await self.get_team(query.team_id)
        if team and query.user_id:
            user_perm = team.get_permission(query.user_id)
            items = [
                item for item in items
                if item.is_public or
                   query.user_id in item.allowed_members or
                   user_perm == TeamPermission.ADMIN
            ]

        # 关键词过滤
        if query.keywords:
            items = [
                item for item in items
                if any(
                    kw.lower() in str(item.content).lower() or
                    kw.lower() in item.title.lower()
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

        return {
            "items": [item.to_dict() for item in items],
            "total": len(items),
            "retrieval_time_ms": retrieval_time,
        }

    async def get_memory(self, memory_id: str) -> Optional[TeamMemory]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM team_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    async def update_memory(self, memory: TeamMemory) -> bool:
        memory.version += 1
        memory.updated_at = datetime.now()
        return await self.store_memory(memory)

    async def delete_memory(self, memory_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM team_memories WHERE id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    async def get_team_memory_count(self, team_id: str) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM team_memories WHERE team_id = ?",
                (team_id,),
            ).fetchone()
            return row["cnt"] if row else 0

    # ─── 内部辅助方法 ────────────────────────────────────

    def _row_to_team(self, row: sqlite3.Row) -> Team:
        members_data = json.loads(row["members"])
        members = [
            TeamMember(
                user_id=m["user_id"],
                role=m.get("role", "member"),
                permission=TeamPermission(m.get("permission", "view")),
                joined_at=datetime.fromisoformat(m["joined_at"]),
            )
            for m in members_data
        ]
        return Team(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            department=row["department"],
            members=members,
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            tags=json.loads(row["tags"]),
            is_active=bool(row["is_active"]),
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> TeamMemory:
        return TeamMemory(
            id=row["id"],
            team_id=row["team_id"],
            title=row["title"],
            memory_type=TeamMemoryType(row["memory_type"]),
            content=json.loads(row["content"]),
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_by=row["updated_by"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            tags=json.loads(row["tags"]),
            keywords=json.loads(row["keywords"]),
            importance=row["importance"],
            is_public=bool(row["is_public"]),
            allowed_members=json.loads(row["allowed_members"]),
            version=row["version"],
            metadata=json.loads(row["metadata"]),
        )