"""
认知记忆模块 - 管理控制台后端 API

提供管理后台所需的全部数据接口，支持:
- 记忆图谱数据查询
- 用户画像雷达图数据
- 行为模式热力图数据
- 系统运行状态监控
- 数据自动刷新（5分钟间隔）
- 后台定时数据模拟（持续产生新数据）
"""

from __future__ import annotations

import json
import os
import random
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_STATIC_DIR = Path(__file__).parent / "static"

# 尝试导入 FastAPI 和依赖
try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# 尝试导入内部模块
try:
    from ..core.memory_manager import MemoryManager
    from ..models.memory import SceneContext
    HAS_MEMORY_MANAGER = True
except ImportError:
    HAS_MEMORY_MANAGER = False


# ─── 后台数据模拟器 ────────────────────────────────────

# 模拟数据素材池
_SIM_WEATHER = ["sunny", "cloudy", "rainy", "windy", "foggy", "snowy"]
_SIM_TRAFFIC = ["smooth", "moderate", "congested"]
_SIM_TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
_SIM_LOCATIONS = ["home", "office", "road", "gym", "mall", "park"]
_SIM_INTENTS = [
    "navigate", "play_music", "adjust_temperature", "switch_mode",
    "find_parking", "check_weather", "call_contact", "safety_alert",
]
_SIM_MUSIC_GENRES = ["pop", "rock", "jazz", "classical", "electronic", "hiphop", "lofi"]
_SIM_TEMP_VALUES = [21, 22, 23, 24, 25, 26, 27, 28]
_SIM_DRIVE_MODES = ["eco", "comfort", "sport", "normal"]

# 用户ID → 姓名映射
_USER_NAMES = {
    "001": "张明", "002": "李芳", "003": "王建国", "004": "陈晓",
    "005": "刘洋", "006": "赵丽", "007": "孙磊", "008": "周婷",
    "009": "吴强", "010": "郑雪", "011": "冯伟", "012": "褚琳",
    "013": "蒋涛", "014": "沈月", "015": "韩飞", "016": "杨华",
    "017": "朱敏", "018": "秦刚", "019": "许晴", "020": "何琳",
}


class BackgroundDataSimulator:
    """后台数据模拟器 - 定时生成模拟交互数据

    每隔 interval 秒，为随机用户生成一条模拟交互记录，
    并根据交互更新行为模式，使前端图谱/仪表盘持续变化。
    """

    def __init__(
        self,
        memory_manager,
        interval: int = 60,
        online_ttl: int = 300,
    ):
        self._mgr = memory_manager
        self._interval = interval
        self._online_ttl = online_ttl  # 在线判定时间窗口(秒)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._sim_count = 0
        self._active_users: dict[str, datetime] = {}  # user_id → 最后活跃时间

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())
        print(f"[DataSimulator] 启动, 间隔 {self._interval}s")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print(f"[DataSimulator] 已停止, 共生成 {self._sim_count} 条数据")

    @property
    def sim_count(self) -> int:
        return self._sim_count

    @property
    def online_users(self) -> list[dict]:
        """返回当前在线用户列表（最近 _online_ttl 秒内有交互）"""
        now = datetime.now()
        result = []
        for uid, last_active in self._active_users.items():
            if (now - last_active).total_seconds() <= self._online_ttl:
                result.append({
                    "user_id": uid,
                    "name": _USER_NAMES.get(uid, uid),
                    "last_active": last_active.isoformat(),
                    "seconds_ago": round((now - last_active).total_seconds()),
                })
        result.sort(key=lambda x: x["seconds_ago"])
        return result

    async def _run(self):
        while self._running:
            try:
                await self._generate_one()
                self._sim_count += 1
            except Exception as e:
                print(f"[DataSimulator] 生成失败: {e}")
            await asyncio.sleep(self._interval)

    async def _generate_one(self):
        from ..models.memory import InteractionRecord, SceneContext, BehaviorPattern

        now = datetime.now()
        user_id = f"{random.randint(1, 20):03d}"

        # 记录活跃用户
        self._active_users[user_id] = now
        intent = random.choice(_SIM_INTENTS)

        # 根据意图生成不同的交互
        if intent == "navigate":
            destinations = ["中关村", "国贸", "五道口", "望京", "三里屯", "西二旗"]
            dest = random.choice(destinations)
            raw_input = random.choice([
                f"导航到{dest}", f"去{dest}", f"带我去{dest}",
            ])
            system_response = {"action": "navigate", "destination": dest}

        elif intent == "play_music":
            genre = random.choice(_SIM_MUSIC_GENRES)
            raw_input = random.choice([
                f"播放{genre}音乐", f"来点{genre}", f"我想听{genre}",
            ])
            system_response = {"action": "play", "genre": genre}

        elif intent == "adjust_temperature":
            temp = random.choice(_SIM_TEMP_VALUES)
            raw_input = random.choice([
                f"温度调到{temp}度", f"有点冷", f"太热了",
            ])
            system_response = {"action": "set_temp", "value": temp}

        elif intent == "switch_mode":
            mode = random.choice(_SIM_DRIVE_MODES)
            raw_input = f"切换到{mode}模式"
            system_response = {"action": "switch_mode", "mode": mode}

        elif intent == "find_parking":
            raw_input = "找附近停车场"
            system_response = {"action": "find_parking", "nearby": "500m"}

        elif intent == "check_weather":
            raw_input = random.choice(["今天天气怎么样", "会下雨吗"])
            system_response = {"action": "weather", "forecast": random.choice(_SIM_WEATHER)}

        elif intent == "call_contact":
            raw_input = "打电话给XX"
            system_response = {"action": "call", "contact": "XX"}

        else:  # safety_alert
            alert = random.choice(["碰撞预警", "车道偏离", "疲劳驾驶提醒", "超速警告"])
            raw_input = alert
            system_response = {"action": "alert", "type": alert}

        record = InteractionRecord(
            user_id=user_id,
            interaction_type=random.choice(["voice_command", "touch_input", "automatic"]),
            intent=intent,
            raw_input=raw_input,
            scene_context=SceneContext(
                time_of_day=random.choice(_SIM_TIME_SLOTS),
                location_type=random.choice(_SIM_LOCATIONS),
                weather=random.choice(_SIM_WEATHER),
                traffic_condition=random.choice(_SIM_TRAFFIC),
            ),
            system_response=system_response,
            response_time_ms=random.randint(80, 400),
            was_successful=random.random() > 0.05,
            timestamp=now,
        )

        await self._mgr._interaction_store.record(record)

        # 尝试更新行为模式（增量刷新置信度）
        try:
            patterns = await self._mgr._pattern_store.get_patterns(user_id, intent)
            if patterns:
                # 更新现有模式的 last_observed
                p = patterns[0]
                p.last_observed = now
                p.occurrence_count += 1
                p.confidence = min(0.98, p.confidence + random.uniform(0.001, 0.01))
                await self._mgr._pattern_store.save_pattern(p)
        except Exception:
            pass  # 模式更新不影响交互记录


# ─── 创建应用 ──────────────────────────────────────────

def create_admin_app(memory_manager=None) -> "FastAPI":
    """创建管理控制台 FastAPI 应用"""
    if not HAS_FASTAPI:
        raise ImportError(
            "需要安装 FastAPI: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="认知记忆管理控制台",
        description="智能座舱认知记忆模块管理后台",
        version="1.0.0",
    )

    # 静态文件
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # 存储管理器引用
    app.state.memory_manager = memory_manager

    # ─── 页面路由 ───────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """管理控制台首页"""
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>管理控制台 - 静态文件未找到</h1>", status_code=404)

    # ─── 数据 API ───────────────────────────────────────

    @app.get("/api/admin/graph")
    async def get_memory_graph(
        user_id: str = Query(default=""),
        limit: int = Query(default=50, le=200),
    ):
        """获取记忆图谱数据

        返回节点和边数据，用于前端可视化渲染。
        支持 1000+ 节点的流畅渲染。
        """
        mgr = app.state.memory_manager
        if mgr is None:
            return {"nodes": [], "edges": [], "error": "MemoryManager 未初始化"}

        nodes = []
        edges = []

        try:
            # 获取行为模式作为节点
            if user_id:
                patterns = await mgr.get_behavior_patterns(user_id)
            else:
                patterns = await mgr.get_all_behavior_patterns(limit)

            for pattern in patterns[:limit]:
                node_id = pattern.id
                nodes.append({
                    "id": node_id,
                    "label": pattern.pattern_name,
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "size": max(5, pattern.confidence * 20),
                    "group": pattern.pattern_type,
                })

                # 为每个模式创建与上下文键的边
                for ctx_key in pattern.related_context_keys:
                    if ctx_key not in {n["id"] for n in nodes}:
                        nodes.append({
                            "id": ctx_key,
                            "label": ctx_key,
                            "type": "context",
                            "confidence": 0.5,
                            "size": 8,
                            "group": "context",
                        })
                    edges.append({
                        "from": node_id,
                        "to": ctx_key,
                        "value": pattern.confidence,
                    })
        except Exception as e:
            return {"nodes": [], "edges": [], "error": str(e)}

        return {"nodes": nodes, "edges": edges}

    @app.get("/api/admin/radar/{user_id}")
    async def get_radar_data(user_id: str):
        """获取用户画像雷达图数据

        返回多维度特征分布数据，用于雷达图渲染。
        """
        mgr = app.state.memory_manager
        if mgr is None:
            return {"labels": [], "datasets": [], "error": "MemoryManager 未初始化"}

        try:
            profile = await mgr.get_user_profile(user_id)
            if profile is None:
                return {"labels": [], "datasets": [], "error": f"用户 {user_id} 画像不存在"}

            labels = [
                "温度偏好",
                "驾驶模式",
                "音乐偏好",
                "交互频率",
                "路线固定性",
                "场景多样性",
                "满意度",
                "反馈活跃度",
            ]

            datasets = [{
                "label": f"用户 {user_id}",
                "data": [
                    min(1.0, profile.confidence_score * 0.9),
                    min(1.0, 0.7 if profile.driving_mode_preference else 0.3),
                    min(1.0, len(profile.music_preferences) / 10),
                    min(1.0, 0.5),
                    min(1.0, len(profile.frequent_routes) / 10),
                    min(1.0, len(profile.active_hours) / 5),
                    profile.confidence_score,
                    min(1.0, profile.data_points_count / 100),
                ],
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 2,
            }]

            return {"labels": labels, "datasets": datasets}
        except Exception as e:
            return {"labels": [], "datasets": [], "error": str(e)}

    @app.get("/api/admin/heatmap/{user_id}")
    async def get_heatmap_data(
        user_id: str,
        days: int = Query(default=7, le=30),
    ):
        """获取行为模式热力图数据

        返回按时间和类型维度的行为频率数据。
        """
        mgr = app.state.memory_manager
        if mgr is None:
            return {"data": [], "x_labels": [], "y_labels": [], "error": "MemoryManager 未初始化"}

        try:
            patterns = await mgr.get_behavior_patterns(user_id)

            # 构建热力图矩阵
            hours = list(range(24))
            pattern_types = list(set(p.pattern_type for p in patterns))

            if not pattern_types:
                pattern_types = ["route", "temperature", "media", "time", "interaction"]

            # 初始化矩阵
            matrix = [[0 for _ in hours] for _ in pattern_types]

            for pattern in patterns:
                type_idx = pattern_types.index(pattern.pattern_type) if pattern.pattern_type in pattern_types else 0
                hour = pattern.last_observed.hour if pattern.last_observed else 12
                matrix[type_idx][hour] += pattern.occurrence_count

            return {
                "data": matrix,
                "x_labels": [f"{h:02d}:00" for h in hours],
                "y_labels": pattern_types,
            }
        except Exception as e:
            return {"data": [], "x_labels": [], "y_labels": [], "error": str(e)}

    @app.get("/api/admin/status")
    async def get_system_status():
        """获取系统运行状态

        包含资源占用、任务进度、存储统计等监控数据。
        数据每5分钟自动刷新，支持手动刷新。
        """
        mgr = app.state.memory_manager
        if mgr is None:
            return {"error": "MemoryManager 未初始化"}

        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
        except ImportError:
            cpu_percent = 0
            memory = type("mem", (), {"percent": 0, "total": 0, "used": 0})()
            disk = type("disk", (), {"percent": 0, "total": 0, "used": 0})()

        try:
            stats = await mgr.get_stats()
        except Exception:
            stats = {}

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_total_gb": round(memory.total / (1024**3), 1),
                "memory_used_gb": round(memory.used / (1024**3), 1),
                "disk_percent": disk.percent,
            },
            "memory_system": {
                "short_term_count": stats.get("short_term_memories", 0),
                "short_term_capacity": stats.get("short_term_capacity", 500),
                "usage_percent": round(
                    stats.get("short_term_memories", 0) / max(1, stats.get("short_term_capacity", 500)) * 100, 1
                ),
            },
            "status": "running",
            "uptime_seconds": getattr(app.state, "start_time", 0) and (
                datetime.now() - app.state.start_time
            ).total_seconds() or 0,
        }

    @app.get("/api/admin/users")
    async def get_users_summary():
        """获取用户概览数据"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"users": [], "error": "MemoryManager 未初始化"}

        # 从反馈统计获取活跃用户
        try:
            feedback_stats = await mgr.get_feedback_stats(days=30)
        except Exception:
            feedback_stats = {}

        return {
            "total_feedback": feedback_stats.get("total", 0),
            "like_rate": feedback_stats.get("like_rate", 0),
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/api/admin/simulator")
    async def get_simulator_status():
        """获取数据模拟器状态"""
        sim = getattr(app.state, "simulator", None)
        if sim is None:
            return {"running": False, "sim_count": 0}
        return {
            "running": sim._running,
            "sim_count": sim.sim_count,
            "interval_seconds": sim._interval,
        }

    @app.get("/api/admin/online-users")
    async def get_online_users():
        """获取当前在线用户"""
        sim = getattr(app.state, "simulator", None)
        if sim is None:
            return {"online": 0, "users": []}
        users = sim.online_users
        return {"online": len(users), "users": users}

    # ─── FE-1: 团队管理 API ──────────────────────────────────

    @app.get("/api/admin/teams/{user_id}")
    async def get_user_teams(user_id: str):
        """获取用户所属团队列表"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"teams": [], "error": "MemoryManager 未初始化"}

        try:
            from ..storage.team_store import TeamStore
            team_store = TeamStore(mgr._long_term._db_path)
            teams = await team_store.get_user_teams(user_id)
            return {"teams": [t.to_dict() for t in teams]}
        except Exception as e:
            return {"teams": [], "error": str(e)}

    @app.post("/api/admin/teams")
    async def admin_create_team(data: dict):
        """创建团队"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"success": False, "error": "MemoryManager 未初始化"}

        try:
            from ..models.team_memory import Team, TeamMember, TeamPermission
            from ..storage.team_store import TeamStore

            team_store = TeamStore(mgr._long_term._db_path)
            members = [TeamMember(
                user_id=m["user_id"],
                role=m.get("role", "member"),
                permission=TeamPermission(m.get("permission", "view")),
            ) for m in data.get("members", [])]

            team = Team(
                name=data["name"],
                description=data.get("description", ""),
                department=data.get("department", ""),
                members=members,
                created_by=data.get("created_by", ""),
            )
            await team_store.create_team(team)
            return {"success": True, "team": team.to_dict()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/admin/teams/{team_id}/memories")
    async def get_team_memories(team_id: str, user_id: str = "", max_results: int = 50):
        """查询团队记忆"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"items": [], "total": 0, "error": "MemoryManager 未初始化"}

        try:
            from ..models.team_memory import TeamMemoryQuery
            from ..storage.team_store import TeamStore

            team_store = TeamStore(mgr._long_term._db_path)
            query = TeamMemoryQuery(
                team_id=team_id,
                user_id=user_id,
                max_results=max_results,
            )
            result = await team_store.query_memories(query)
            return result
        except Exception as e:
            return {"items": [], "total": 0, "error": str(e)}

    @app.post("/api/admin/teams/{team_id}/memories")
    async def admin_create_team_memory(team_id: str, data: dict):
        """创建团队记忆"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"success": False, "error": "MemoryManager 未初始化"}

        try:
            from ..models.team_memory import TeamMemory, TeamMemoryType
            from ..storage.team_store import TeamStore

            team_store = TeamStore(mgr._long_term._db_path)
            memory = TeamMemory(
                team_id=team_id,
                title=data.get("title", ""),
                memory_type=TeamMemoryType(data.get("memory_type", "general")),
                content=data.get("content", {}),
                created_by=data.get("created_by", ""),
                tags=data.get("tags", []),
                keywords=data.get("keywords", []),
                importance=data.get("importance", 3),
                is_public=data.get("is_public", True),
                allowed_members=data.get("allowed_members", []),
            )
            await team_store.store_memory(memory)
            return {"success": True, "memory": memory.to_dict()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── FE-4: 文档上下文 API ────────────────────────────────

    @app.get("/api/admin/documents/{user_id}")
    async def get_user_documents(user_id: str, limit: int = 20):
        """获取用户最近文档上下文"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"documents": [], "error": "MemoryManager 未初始化"}

        try:
            # 从交互记录中提取文档上下文信息
            from ..models.memory import MemoryType
            result = await mgr.query_memories(
                user_id=user_id,
                max_results=limit,
                sort_by="recent",
            )
            # 模拟文档上下文数据（实际应由文档跟踪模块提供）
            documents = []
            for item in result.items[:limit]:
                if hasattr(item, 'content') and isinstance(item.content, dict):
                    doc_info = item.content.get("document_context")
                    if doc_info:
                        documents.append(doc_info)
            return {"documents": documents}
        except Exception as e:
            return {"documents": [], "error": str(e)}

    @app.get("/api/admin/documents/{user_id}/recent")
    async def get_recent_documents(user_id: str):
        """获取用户最近编辑的5个文档"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"documents": [], "error": "MemoryManager 未初始化"}

        try:
            # 模拟文档数据（演示用）
            import random
            from datetime import datetime, timedelta

            doc_templates = [
                {"file_name": "需求文档_v2.docx", "format": "docx", "summary": "智能座舱AI系统需求规格说明"},
                {"file_name": "架构设计文档.pdf", "format": "pdf", "summary": "认知记忆模块架构设计v2.1"},
                {"file_name": "接口规范.docx", "format": "docx", "summary": "API接口规范文档"},
                {"file_name": "测试报告.xlsx", "format": "xlsx", "summary": "Sprint 12 测试报告"},
                {"file_name": "会议纪要.txt", "format": "txt", "summary": "2026-07-15 架构评审会议纪要"},
            ]

            now = datetime.now()
            documents = []
            for i, doc in enumerate(doc_templates):
                edit_time = now - timedelta(hours=i * 2)
                documents.append({
                    "id": f"doc_{i+1}",
                    "file_name": doc["file_name"],
                    "file_format": doc["format"],
                    "content_summary": doc["summary"],
                    "last_accessed_at": edit_time.isoformat(),
                    "edit_count": random.randint(1, 15),
                    "total_edit_time": random.randint(60, 3600),
                    "keywords": ["AI", "座舱", "认知"][:random.randint(1, 3)],
                    "associated_sessions": [f"session_{random.randint(1, 100)}" for _ in range(random.randint(1, 3))],
                })

            return {"documents": documents}
        except Exception as e:
            return {"documents": [], "error": str(e)}

    # ─── FE-7: 数据导出 API ──────────────────────────────────

    @app.post("/api/admin/export")
    async def admin_export_data(data: dict):
        """导出用户记忆数据"""
        mgr = app.state.memory_manager
        if mgr is None:
            return {"success": False, "error": "MemoryManager 未初始化"}

        try:
            from ..api.routes import MemoryAPI, ExportDataRequest
            api = MemoryAPI(mgr)
            request = ExportDataRequest(
                user_id=data.get("user_id", ""),
                format=data.get("format", "json"),
                categories=data.get("categories", []),
                include_metadata=data.get("include_metadata", True),
            )
            result = await api.export_data(request)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── 生命周期管理 ───────────────────────────────────

    @app.on_event("startup")
    async def on_startup():
        mgr = app.state.memory_manager
        if mgr is not None:
            simulator = BackgroundDataSimulator(mgr, interval=60)
            app.state.simulator = simulator
            await simulator.start()
        app.state.start_time = datetime.now()

    @app.on_event("shutdown")
    async def on_shutdown():
        sim = getattr(app.state, "simulator", None)
        if sim is not None:
            await sim.stop()

    return app


# 模块级 app 实例，用于 uvicorn 直接启动: uvicorn cognitive_memory.admin.server:app
if HAS_MEMORY_MANAGER:
    _default_manager = MemoryManager(
        db_path=str(Path(__file__).parent.parent.parent / "cognitive_memory.db")
    )
else:
    _default_manager = None

app = create_admin_app(_default_manager)


def main():
    """启动管理控制台服务"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()