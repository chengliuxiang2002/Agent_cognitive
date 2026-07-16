"""
认知记忆模块 - 测试共享夹具
"""

import pytest
from cognitive_memory.core import MemoryManager


@pytest.fixture
def memory_manager(tmp_path):
    """创建测试用的 MemoryManager，使用临时目录"""
    db_path = str(tmp_path / "test_memory.db")
    return MemoryManager(db_path=db_path)