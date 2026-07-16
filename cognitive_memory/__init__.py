"""
智能座舱Agent - 认知记忆模块 (Cognitive Memory Module)

本模块负责处理、存储和检索用户相关信息，以支持个性化交互和上下文感知功能。
包含以下核心子模块:
- models:    记忆数据模型定义
- storage:   混合存储架构（短期/长期记忆）
- learner:   记忆学习算法与衰减机制
- core:      核心业务逻辑（记忆管理、用户画像、上下文引擎）
- api:       对外接口层
"""

__version__ = "1.0.0"
__author__ = "Intelligent Cockpit Team"