"""
认知记忆模块 - 管理控制台

提供 Web 管理后台系统，包含:
- 记忆图谱可视化
- 用户画像雷达图
- 行为模式热力图
- 系统运行状态监控

技术栈: FastAPI + 原生 HTML/CSS/JS (Chart.js)

启动方式:
    python -m cognitive_memory.admin.server
    或
    uvicorn cognitive_memory.admin.server:app --port 8080
"""