"""
认知记忆模块 - RBAC 权限管理 (SC-2)

基于角色的访问控制 (Role-Based Access Control):
- 角色体系: admin (管理员), manager (经理), employee (普通员工)
- 操作权限: read_own, read_team, write, delete
- @require_permission 装饰器用于 API 层权限控制
- 用户-角色-权限映射关系的数据表
"""

from __future__ import annotations

import functools
import json
import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Role(Enum):
    """角色定义 (SC-2)"""
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"


class Permission(Enum):
    """操作权限定义 (SC-2)"""
    READ_OWN = "read_own"       # 查看自己的数据
    READ_TEAM = "read_team"     # 查看团队数据
    WRITE = "write"             # 写入数据
    DELETE = "delete"           # 删除数据


# SC-2: 角色-权限映射表
# admin: 拥有所有权限
# manager: 可以查看团队数据、写入和删除
# employee: 仅可查看自己的数据和写入
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {
        Permission.READ_OWN,
        Permission.READ_TEAM,
        Permission.WRITE,
        Permission.DELETE,
    },
    Role.MANAGER: {
        Permission.READ_OWN,
        Permission.READ_TEAM,
        Permission.WRITE,
        Permission.DELETE,
    },
    Role.EMPLOYEE: {
        Permission.READ_OWN,
        Permission.WRITE,
    },
}


class PermissionError(Exception):
    """权限错误"""
    pass


class PermissionManager:
    """SC-2: 权限验证与管理

    负责:
    1. 用户-角色映射管理
    2. 权限验证
    3. 角色分配与变更
    """

    def __init__(self):
        # 用户-角色映射: {user_id: Role}
        self._user_roles: dict[str, Role] = {}
        # 用户-团队映射: {user_id: team_id}
        self._user_teams: dict[str, str] = {}
        # 团队成员: {team_id: set[user_id]}
        self._team_members: dict[str, set[str]] = {}

    # ─── 角色管理 ───────────────────────────────────────

    def assign_role(self, user_id: str, role: Role):
        """为用户分配角色"""
        self._user_roles[user_id] = role
        logger.info(f"Assigned role {role.value} to user {user_id}")

    def get_role(self, user_id: str) -> Role:
        """获取用户角色 (默认 employee)"""
        return self._user_roles.get(user_id, Role.EMPLOYEE)

    def revoke_role(self, user_id: str):
        """撤销用户角色 (回退为默认 employee)"""
        self._user_roles.pop(user_id, None)

    # ─── 团队管理 ───────────────────────────────────────

    def assign_to_team(self, user_id: str, team_id: str):
        """将用户分配到团队"""
        self._user_teams[user_id] = team_id
        if team_id not in self._team_members:
            self._team_members[team_id] = set()
        self._team_members[team_id].add(user_id)

    def get_user_team(self, user_id: str) -> Optional[str]:
        """获取用户所属团队"""
        return self._user_teams.get(user_id)

    def get_team_members(self, team_id: str) -> set[str]:
        """获取团队成员"""
        return self._team_members.get(team_id, set())

    def is_same_team(self, user_id: str, target_user_id: str) -> bool:
        """判断两个用户是否在同一团队"""
        team_a = self._user_teams.get(user_id)
        team_b = self._user_teams.get(target_user_id)
        return team_a is not None and team_a == team_b

    # ─── 权限验证 ───────────────────────────────────────

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """检查用户是否拥有指定权限"""
        role = self.get_role(user_id)
        allowed = ROLE_PERMISSIONS.get(role, set())
        return permission in allowed

    def check_permission(self, user_id: str, permission: Permission):
        """验证权限，无权限时抛出 PermissionError"""
        if not self.has_permission(user_id, permission):
            raise PermissionError(
                f"User {user_id} does not have permission: {permission.value}"
            )

    def check_read_access(
        self, requester_id: str, target_user_id: str
    ):
        """检查读取权限

        - 用户可读取自己的数据 (read_own)
        - 同团队可读取 (read_team)
        """
        if requester_id == target_user_id:
            self.check_permission(requester_id, Permission.READ_OWN)
            return

        if self.has_permission(requester_id, Permission.READ_TEAM):
            if self.is_same_team(requester_id, target_user_id):
                return

        raise PermissionError(
            f"User {requester_id} cannot access data of user {target_user_id}"
        )

    def check_write_access(self, user_id: str):
        """检查写入权限"""
        self.check_permission(user_id, Permission.WRITE)

    def check_delete_access(self, user_id: str, target_user_id: Optional[str] = None):
        """检查删除权限

        - 用户可删除自己的数据 (read_own + delete)
        - admin 可删除任何数据
        """
        self.check_permission(user_id, Permission.DELETE)
        if target_user_id and target_user_id != user_id:
            if self.get_role(user_id) != Role.ADMIN:
                raise PermissionError(
                    f"User {user_id} cannot delete data of user {target_user_id}"
                )

    # ─── 序列化 ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "user_roles": {k: v.value for k, v in self._user_roles.items()},
            "user_teams": self._user_teams,
            "team_members": {k: list(v) for k, v in self._team_members.items()},
        }

    def from_dict(self, data: dict[str, Any]):
        """从字典恢复"""
        self._user_roles = {k: Role(v) for k, v in data.get("user_roles", {}).items()}
        self._user_teams = data.get("user_teams", {})
        self._team_members = {k: set(v) for k, v in data.get("team_members", {}).items()}


# ═══════════════════════════════════════════════════════════════════════════════
# SC-2: @require_permission 装饰器
# ═══════════════════════════════════════════════════════════════════════════════


def require_permission(
    permission: Permission,
    permission_manager_provider: Optional[Callable[[], PermissionManager]] = None,
):
    """SC-2: API 层权限控制装饰器

    用法:
        @require_permission(Permission.READ_OWN, get_permission_manager)
        async def get_profile(self, user_id: str) -> ApiResponse:
            ...

    如果未提供 permission_manager_provider，则从函数所在类的
    _permission_manager 属性获取。

    Args:
        permission: 所需权限
        permission_manager_provider: 可选的 PermissionManager 提供者函数
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取 PermissionManager 实例
            pm = None
            if permission_manager_provider:
                pm = permission_manager_provider()
            elif args:
                # 从实例的 _permission_manager 属性获取
                instance = args[0]
                pm = getattr(instance, "_permission_manager", None)

            if pm is None:
                raise PermissionError("PermissionManager not available")

            # 尝试从参数中获取 user_id
            # 优先从 kwargs 获取，其次从 args 的第二个参数获取
            user_id = kwargs.get("user_id")
            if user_id is None and len(args) > 1:
                # 第二个参数可能是 request 对象或 user_id 字符串
                arg2 = args[1]
                if isinstance(arg2, str):
                    user_id = arg2
                elif hasattr(arg2, "user_id"):
                    user_id = arg2.user_id

            if user_id is None:
                raise PermissionError("Cannot determine user_id from request")

            pm.check_permission(user_id, permission)
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_own_or_team_access(
    permission_manager_provider: Optional[Callable[[], PermissionManager]] = None,
):
    """SC-2: 检查用户只能访问自己或团队的数据

    用于需要区分 read_own 和 read_team 的场景。
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            pm = None
            if permission_manager_provider:
                pm = permission_manager_provider()
            elif args:
                instance = args[0]
                pm = getattr(instance, "_permission_manager", None)

            if pm is None:
                raise PermissionError("PermissionManager not available")

            requester_id = kwargs.get("requester_user_id") or kwargs.get("user_id")
            if requester_id is None and len(args) > 1:
                arg2 = args[1]
                if isinstance(arg2, str):
                    requester_id = arg2
                elif hasattr(arg2, "user_id"):
                    requester_id = arg2.user_id

            target_user_id = kwargs.get("target_user_id") or kwargs.get("user_id")
            if target_user_id is None:
                target_user_id = requester_id

            if requester_id is None:
                raise PermissionError("Cannot determine user_id from request")

            pm.check_read_access(requester_id, target_user_id)
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SC-2: 权限数据持久化表结构
# ═══════════════════════════════════════════════════════════════════════════════

# SQL 表结构定义 (用于 init_db 时创建)
PERMISSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_roles (
    user_id TEXT PRIMARY KEY,
    role TEXT NOT NULL DEFAULT 'employee',
    assigned_at TEXT NOT NULL,
    assigned_by TEXT NOT NULL DEFAULT 'system',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS user_teams (
    user_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (user_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_user_teams_team_id
    ON user_teams(team_id);

CREATE TABLE IF NOT EXISTS permission_audit (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    resource TEXT NOT NULL,
    result TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ip_address TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_permission_audit_user_id
    ON permission_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_permission_audit_timestamp
    ON permission_audit(timestamp);
"""


class PermissionStore:
    """SC-2: 权限数据持久化存储

    将用户角色、团队映射关系持久化到 SQLite 数据库。
    """

    def __init__(self, db_path: str):
        import sqlite3
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """初始化权限相关表"""
        self._conn.executescript(PERMISSION_TABLE_SQL)
        self._conn.commit()

    def load_permission_manager(self) -> PermissionManager:
        """从数据库加载权限数据"""
        pm = PermissionManager()

        # 加载用户角色
        rows = self._conn.execute("SELECT user_id, role FROM user_roles").fetchall()
        for row in rows:
            pm._user_roles[row["user_id"]] = Role(row["role"])

        # 加载用户团队
        rows = self._conn.execute("SELECT user_id, team_id FROM user_teams").fetchall()
        for row in rows:
            user_id = row["user_id"]
            team_id = row["team_id"]
            pm._user_teams[user_id] = team_id
            if team_id not in pm._team_members:
                pm._team_members[team_id] = set()
            pm._team_members[team_id].add(user_id)

        return pm

    def save_role(self, user_id: str, role: Role, assigned_by: str = "system"):
        """保存用户角色"""
        from datetime import datetime
        self._conn.execute(
            """INSERT OR REPLACE INTO user_roles
               (user_id, role, assigned_at, assigned_by)
               VALUES (?, ?, ?, ?)""",
            (user_id, role.value, datetime.now().isoformat(), assigned_by),
        )
        self._conn.commit()

    def save_team_assignment(self, user_id: str, team_id: str):
        """保存用户团队分配"""
        from datetime import datetime
        self._conn.execute(
            """INSERT OR REPLACE INTO user_teams
               (user_id, team_id, joined_at)
               VALUES (?, ?, ?)""",
            (user_id, team_id, datetime.now().isoformat()),
        )
        self._conn.commit()

    def audit_permission_check(
        self,
        user_id: str,
        permission: str,
        resource: str,
        result: str,
        ip_address: str = "",
    ):
        """记录权限检查审计日志"""
        import uuid
        from datetime import datetime

        self._conn.execute(
            "INSERT INTO permission_audit (id, user_id, permission, resource, result, timestamp, ip_address) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                user_id,
                permission,
                resource,
                result,
                datetime.now().isoformat(),
                ip_address,
            ),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# 全局默认权限管理器实例
_default_permission_manager: Optional[PermissionManager] = None


def get_default_permission_manager() -> PermissionManager:
    """获取全局默认权限管理器"""
    global _default_permission_manager
    if _default_permission_manager is None:
        _default_permission_manager = PermissionManager()
    return _default_permission_manager


def set_default_permission_manager(pm: PermissionManager):
    """设置全局默认权限管理器"""
    global _default_permission_manager
    _default_permission_manager = pm