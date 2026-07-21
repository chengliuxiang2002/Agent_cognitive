"""
认知记忆模块 - 数据安全与权限管理测试 (SC-1 ~ SC-7)
"""

import json
import os
import time
from datetime import datetime, timedelta

import pytest

from cognitive_memory.core.privacy import (
    MemoryEncryption,
    PrivacyManager,
    TieredEncryption,
    AuditLogger,
)
from cognitive_memory.core.permission import (
    PermissionManager,
    PermissionStore,
    Permission,
    Role,
    PermissionError,
    require_permission,
    require_own_or_team_access,
    ROLE_PERMISSIONS,
    get_default_permission_manager,
    set_default_permission_manager,
)
from cognitive_memory.models.memory import MemoryImportance, MemoryItem, MemoryType
from cognitive_memory.api.routes import (
    AuthToken,
    SSOAuthService,
    TokenAuthMiddleware,
    UnauthorizedError,
    TokenExpiredError,
    MemoryAPI,
    ApiResponse,
    RecordInteractionRequest,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SC-1: 强制加密测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestMandatoryEncryption:
    """SC-1: 强制加密 - 移除 XOR 降级，使用 cryptography 库"""

    def test_encrypt_decrypt_roundtrip(self):
        """加密解密往返测试"""
        enc = MemoryEncryption()
        original = {"user_id": "test_001", "name": "张三", "phone": "13800138000"}
        encrypted = enc.encrypt(original)
        assert "ciphertext" in encrypted
        assert "nonce" in encrypted
        decrypted = enc.decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_different_keys_produce_different_ciphertext(self):
        """不同密钥产生不同密文"""
        enc1 = MemoryEncryption()
        enc2 = MemoryEncryption()
        data = {"key": "value"}
        result1 = enc1.encrypt(data)
        result2 = enc2.encrypt(data)
        assert result1["ciphertext"] != result2["ciphertext"]

    def test_encrypt_same_key_same_data_different_nonce(self):
        """同密钥同数据不同nonce产生不同密文"""
        enc = MemoryEncryption()
        data = {"key": "value"}
        result1 = enc.encrypt(data)
        result2 = enc.encrypt(data)
        assert result1["ciphertext"] != result2["ciphertext"]

    def test_decrypt_with_wrong_data_raises_error(self):
        """解密错误数据抛出异常"""
        enc = MemoryEncryption()
        data = {"key": "value"}
        encrypted = enc.encrypt(data)
        encrypted["ciphertext"] = "invalid_base64"
        with pytest.raises(Exception):
            enc.decrypt(encrypted)

    def test_encrypt_empty_dict(self):
        """加密空字典"""
        enc = MemoryEncryption()
        encrypted = enc.encrypt({})
        decrypted = enc.decrypt(encrypted)
        assert decrypted == {}

    def test_encrypt_complex_data(self):
        """加密复杂嵌套数据"""
        enc = MemoryEncryption()
        data = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "chinese": "中文测试",
        }
        encrypted = enc.encrypt(data)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == data

    def test_no_xor_fallback(self):
        """验证不存在 XOR 降级方案"""
        enc = MemoryEncryption()
        assert not hasattr(enc, "_simple_encrypt"), "XOR fallback should be removed"
        assert not hasattr(enc, "_simple_decrypt"), "XOR fallback should be removed"


# ═══════════════════════════════════════════════════════════════════════════════
# SC-2: RBAC 权限模型测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestRBACPermissionModel:
    """SC-2: RBAC 权限模型"""

    def test_role_permissions_admin(self):
        """admin 拥有所有权限"""
        assert ROLE_PERMISSIONS[Role.ADMIN] == {
            Permission.READ_OWN,
            Permission.READ_TEAM,
            Permission.WRITE,
            Permission.DELETE,
        }

    def test_role_permissions_manager(self):
        """manager 拥有团队读写权限"""
        assert Permission.READ_OWN in ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.READ_TEAM in ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.WRITE in ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.DELETE in ROLE_PERMISSIONS[Role.MANAGER]

    def test_role_permissions_employee(self):
        """employee 仅有读写自己数据的权限"""
        perms = ROLE_PERMISSIONS[Role.EMPLOYEE]
        assert Permission.READ_OWN in perms
        assert Permission.WRITE in perms
        assert Permission.READ_TEAM not in perms
        assert Permission.DELETE not in perms

    def test_assign_role(self):
        """角色分配"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.ADMIN)
        assert pm.get_role("user_001") == Role.ADMIN

    def test_default_role(self):
        """默认角色为 employee"""
        pm = PermissionManager()
        assert pm.get_role("unknown_user") == Role.EMPLOYEE

    def test_revoke_role(self):
        """撤销角色后回退为默认"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.ADMIN)
        pm.revoke_role("user_001")
        assert pm.get_role("user_001") == Role.EMPLOYEE

    def test_has_permission(self):
        """权限检查"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.ADMIN)
        assert pm.has_permission("user_001", Permission.DELETE)
        pm.assign_role("user_002", Role.EMPLOYEE)
        assert not pm.has_permission("user_002", Permission.DELETE)

    def test_check_permission_raises(self):
        """无权限时抛出异常"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.EMPLOYEE)
        with pytest.raises(PermissionError):
            pm.check_permission("user_001", Permission.DELETE)

    def test_team_assignment(self):
        """团队分配"""
        pm = PermissionManager()
        pm.assign_to_team("user_001", "team_A")
        pm.assign_to_team("user_002", "team_A")
        assert pm.get_user_team("user_001") == "team_A"
        assert pm.is_same_team("user_001", "user_002")

    def test_check_read_access_own_data(self):
        """读取自己数据"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.EMPLOYEE)
        pm.check_read_access("user_001", "user_001")  # 不应抛出异常

    def test_check_read_access_team_data(self):
        """读取团队数据"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.MANAGER)
        pm.assign_role("user_002", Role.EMPLOYEE)
        pm.assign_to_team("user_001", "team_A")
        pm.assign_to_team("user_002", "team_A")
        pm.check_read_access("user_001", "user_002")  # 不应抛出异常

    def test_check_read_access_denied(self):
        """跨团队读取被拒绝"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.EMPLOYEE)
        pm.assign_to_team("user_001", "team_A")
        pm.assign_to_team("user_002", "team_B")
        with pytest.raises(PermissionError):
            pm.check_read_access("user_001", "user_002")

    def test_check_delete_access_admin(self):
        """admin 可删除他人数据"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.ADMIN)
        pm.check_delete_access("user_001", "user_002")  # 不应抛出异常

    def test_check_delete_access_denied(self):
        """employee 不可删除他人数据"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.EMPLOYEE)
        with pytest.raises(PermissionError):
            pm.check_delete_access("user_001", "user_002")

    def test_permission_store_persistence(self):
        """权限数据持久化"""
        store = PermissionStore(":memory:")
        store.save_role("user_001", Role.ADMIN)
        store.save_team_assignment("user_001", "team_A")
        pm = store.load_permission_manager()
        assert pm.get_role("user_001") == Role.ADMIN
        assert pm.get_user_team("user_001") == "team_A"
        store.close()

    def test_permission_audit_log(self):
        """权限审计日志"""
        store = PermissionStore(":memory:")
        store.audit_permission_check("user_001", "read_own", "profile", "success")
        store.close()

    def test_serialize_deserialize(self):
        """序列化与反序列化"""
        pm = PermissionManager()
        pm.assign_role("user_001", Role.ADMIN)
        pm.assign_to_team("user_001", "team_A")
        data = pm.to_dict()
        pm2 = PermissionManager()
        pm2.from_dict(data)
        assert pm2.get_role("user_001") == Role.ADMIN
        assert pm2.get_user_team("user_001") == "team_A"


# ═══════════════════════════════════════════════════════════════════════════════
# SC-3: 数据分级存储测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestTieredStorage:
    """SC-3: 数据分级存储"""

    def test_should_encrypt_critical(self):
        """CRITICAL 级别需要加密"""
        te = TieredEncryption()
        assert te.should_encrypt(MemoryImportance.CRITICAL)

    def test_should_encrypt_high(self):
        """HIGH 级别需要加密"""
        te = TieredEncryption()
        assert te.should_encrypt(MemoryImportance.HIGH)

    def test_should_not_encrypt_medium(self):
        """MEDIUM 级别不需要加密"""
        te = TieredEncryption()
        assert not te.should_encrypt(MemoryImportance.MEDIUM)

    def test_should_not_encrypt_low(self):
        """LOW 级别不需要加密"""
        te = TieredEncryption()
        assert not te.should_encrypt(MemoryImportance.LOW)

    def test_should_persist_normal(self):
        """非 TRANSIENT 级别需要持久化"""
        te = TieredEncryption()
        assert te.should_persist(MemoryImportance.CRITICAL)
        assert te.should_persist(MemoryImportance.HIGH)
        assert te.should_persist(MemoryImportance.MEDIUM)
        assert te.should_persist(MemoryImportance.LOW)

    def test_should_not_persist_transient(self):
        """TRANSIENT 级别不需要持久化"""
        te = TieredEncryption()
        assert not te.should_persist(MemoryImportance.TRANSIENT)

    def test_encrypt_if_needed_critical(self):
        """CRITICAL 数据加密后带标记"""
        te = TieredEncryption()
        data = {"secret": "top_secret"}
        result = te.encrypt_if_needed(data, MemoryImportance.CRITICAL)
        assert result["_encrypted"] is True
        assert "ciphertext" in result
        assert "nonce" in result

    def test_encrypt_if_needed_low(self):
        """LOW 数据不加密"""
        te = TieredEncryption()
        data = {"public": "info"}
        result = te.encrypt_if_needed(data, MemoryImportance.LOW)
        assert result["_encrypted"] is False
        assert "public" in result

    def test_decrypt_if_needed(self):
        """解密加密数据"""
        te = TieredEncryption()
        original = {"key": "value"}
        encrypted = te.encrypt_if_needed(original, MemoryImportance.CRITICAL)
        decrypted = te.decrypt_if_needed(encrypted)
        assert decrypted == original

    def test_decrypt_if_needed_plaintext(self):
        """明文数据不解密"""
        te = TieredEncryption()
        data = {"_encrypted": False, "_importance": 2, "key": "value"}
        result = te.decrypt_if_needed(data)
        assert result == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════════════
# SC-4: 审计日志集中化测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditLogger:
    """SC-4: 审计日志集中化"""

    def test_audit_log_structure(self):
        """审计日志结构化 JSON 格式"""
        audit = AuditLogger(log_path="test_audit.log")
        audit.log_access(
            user_id="user_001",
            operation="read",
            data_type="profile",
            requester="system",
            success=True,
            ip_address="192.168.1.1",
        )
        audit.close()
        # 验证日志文件存在且包含 JSON
        assert os.path.exists("test_audit.log")
        with open("test_audit.log", "r") as f:
            content = f.read()
        assert "user_001" not in content  # 匿名化
        assert "read" in content
        assert "profile" in content
        assert "integrity_hash" in content
        os.remove("test_audit.log")

    def test_audit_log_integrity(self):
        """审计日志完整性校验"""
        audit = AuditLogger()
        entry = audit._build_audit_entry(
            user_id="user_001",
            operation="write",
            data_type="memory",
        )
        assert "integrity_hash" in entry
        assert audit.verify_log_integrity(entry)

    def test_audit_log_integrity_tampered(self):
        """篡改日志后完整性验证失败"""
        audit = AuditLogger()
        entry = audit._build_audit_entry(
            user_id="user_001",
            operation="delete",
            data_type="memory",
        )
        entry["operation_type"] = "modified"
        assert not audit.verify_log_integrity(entry)

    def test_audit_log_required_fields(self):
        """审计日志包含必要字段"""
        audit = AuditLogger()
        entry = audit._build_audit_entry(
            user_id="user_001",
            operation="read",
            data_type="profile",
            ip_address="10.0.0.1",
        )
        assert "timestamp" in entry
        assert "operator" in entry
        assert "operation_type" in entry
        assert "target" in entry
        assert "ip_address" in entry
        assert "result" in entry


# ═══════════════════════════════════════════════════════════════════════════════
# SC-5: 数据脱敏增强测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataMasking:
    """SC-5: 数据脱敏增强"""

    def test_mask_phone_number(self):
        """手机号脱敏"""
        data = {"phone_number": "13812345678"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["phone_number"] == "138****5678"

    def test_mask_email(self):
        """邮箱脱敏"""
        data = {"email": "test@example.com"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["email"] == "t***@example.com"

    def test_mask_id_card(self):
        """身份证号脱敏"""
        data = {"id_card": "110101199001011234"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert "*" in result["id_card"]
        assert result["id_card"].startswith("110")

    def test_mask_name(self):
        """姓名脱敏"""
        data = {"name": "张三丰"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["name"] == "张**"

    def test_mask_bank_account(self):
        """银行账号脱敏"""
        data = {"bank_account": "6222021234567890"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["bank_account"] == "****7890"

    def test_mask_license_plate(self):
        """车牌号脱敏"""
        data = {"license_plate": "京A12345"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["license_plate"].startswith("京")
        assert result["license_plate"].endswith("5")

    def test_mask_full_health(self):
        """健康信息完全脱敏"""
        data = {"health_conditions": "高血压"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["health_conditions"] == "***"

    def test_mask_nested_dict(self):
        """递归脱敏嵌套字典"""
        data = {
            "user": {
                "name": "张三",
                "contact": {"phone_number": "13800000000"},
            },
            "public": "info",
        }
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["user"]["name"] == "张*"
        assert result["user"]["contact"]["phone_number"] == "138****0000"
        assert result["public"] == "info"

    def test_mask_list_with_dicts(self):
        """脱敏列表中的字典"""
        data = {
            "contacts": [
                {"name": "张三", "phone_number": "13800000001"},
                {"name": "李四", "phone_number": "13800000002"},
            ]
        }
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["contacts"][0]["name"] == "张*"
        assert result["contacts"][0]["phone_number"] == "138****0001"

    def test_dynamic_add_remove_field(self):
        """动态添加/移除敏感字段"""
        PrivacyManager.add_sensitive_field("custom_secret", "full")
        assert "custom_secret" in PrivacyManager.SENSITIVE_FIELDS
        data = {"custom_secret": "secret_value"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["custom_secret"] == "***"
        PrivacyManager.remove_sensitive_field("custom_secret")
        assert "custom_secret" not in PrivacyManager.SENSITIVE_FIELDS

    def test_mask_short_value(self):
        """短字符串脱敏"""
        data = {"name": "ab"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["name"] == "a*"

    def test_non_sensitive_field_passes_through(self):
        """非敏感字段原样通过"""
        data = {"normal_field": "hello world"}
        result = PrivacyManager.mask_sensitive_data(data)
        assert result["normal_field"] == "hello world"


# ═══════════════════════════════════════════════════════════════════════════════
# SC-6: API 认证集成测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestAPIAuthentication:
    """SC-6: API 认证集成"""

    def test_auth_token_generate(self):
        """Token 生成"""
        auth = AuthToken()
        token = auth.generate_token("user_001")
        assert token is not None
        assert "." in token

    def test_auth_token_validate(self):
        """Token 验证"""
        auth = AuthToken()
        token = auth.generate_token("user_001")
        claims = auth.validate_token(token)
        assert claims["user_id"] == "user_001"
        assert "exp" in claims
        assert "iat" in claims

    def test_auth_token_extract_user_id(self):
        """从 Token 提取 user_id"""
        auth = AuthToken()
        token = auth.generate_token("user_001")
        user_id = auth.extract_user_id(token)
        assert user_id == "user_001"

    def test_auth_token_invalid_signature(self):
        """无效签名验证失败"""
        auth = AuthToken()
        token = auth.generate_token("user_001")
        tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
        with pytest.raises(UnauthorizedError):
            auth.validate_token(tampered)

    def test_auth_token_expired(self):
        """过期 Token 验证失败"""
        import time
        # 创建一个已过期的 token
        auth = AuthToken()
        auth._token_duration_s = -1  # 立即过期
        token = auth.generate_token("user_001")
        time.sleep(0.1)
        with pytest.raises(TokenExpiredError):
            auth.validate_token(token)

    def test_auth_token_refresh(self):
        """Token 刷新"""
        auth = AuthToken()
        auth._refresh_window_s = 3600  # 大窗口确保可刷新
        token = auth.generate_token("user_001")
        new_token = auth.refresh_token(token)
        claims = auth.validate_token(new_token)
        assert claims["user_id"] == "user_001"

    def test_token_should_refresh(self):
        """检测 Token 是否需要刷新"""
        auth = AuthToken()
        # 使用正常的 Duration 和 Refresh Window
        token = auth.generate_token("user_001")
        # 新 token 不应该需要刷新
        assert not auth.should_refresh(token)

    def test_token_should_refresh_expiring(self):
        """即将过期的 Token 需要刷新"""
        auth = AuthToken()
        # 生成一个1小时后过期的token，设置刷新窗口为2小时
        # 这样新token就在刷新窗口内
        auth._token_duration_s = 3600
        auth._refresh_window_s = 7200
        token = auth.generate_token("user_001")
        assert auth.should_refresh(token)

    def test_token_should_not_refresh_new(self):
        """新 Token 不需要刷新"""
        auth = AuthToken()
        auth._token_duration_s = 3600
        auth._refresh_window_s = 300
        token = auth.generate_token("user_001")
        assert not auth.should_refresh(token)

    def test_token_with_extra_claims(self):
        """带有额外 claims 的 Token"""
        auth = AuthToken()
        token = auth.generate_token("user_001", extra_claims={"role": "admin"})
        claims = auth.validate_token(token)
        assert claims["role"] == "admin"

    def test_middleware_authenticate(self):
        """中间件认证"""
        auth = AuthToken()
        middleware = TokenAuthMiddleware(auth)
        token = auth.generate_token("user_001")

        class MockRequest:
            authorization = f"Bearer {token}"

        request = MockRequest()
        import asyncio
        user_id = asyncio.run(middleware.authenticate(request))
        assert user_id == "user_001"

    def test_middleware_missing_header(self):
        """缺少认证头"""
        auth = AuthToken()
        middleware = TokenAuthMiddleware(auth)

        class MockRequest:
            authorization = ""

        request = MockRequest()
        import asyncio
        with pytest.raises(UnauthorizedError):
            asyncio.run(middleware.authenticate(request))

    def test_middleware_invalid_format(self):
        """无效认证头格式"""
        auth = AuthToken()
        middleware = TokenAuthMiddleware(auth)

        class MockRequest:
            authorization = "Basic xxx"

        request = MockRequest()
        import asyncio
        with pytest.raises(UnauthorizedError):
            asyncio.run(middleware.authenticate(request))

    def test_standard_error_response(self):
        """标准错误响应"""
        resp = TokenAuthMiddleware.get_standard_error_response(
            UnauthorizedError("test")
        )
        assert resp["error"] == "UNAUTHORIZED"
        assert resp["error_code"] == 401

        resp = TokenAuthMiddleware.get_standard_error_response(
            TokenExpiredError("test")
        )
        assert resp["error"] == "TOKEN_EXPIRED"
        assert resp["error_code"] == 401

    def test_memory_api_with_auth(self):
        """MemoryAPI 集成认证"""
        from cognitive_memory.core.memory_manager import MemoryManager

        auth = AuthToken()
        manager = MemoryManager(db_path=":memory:")
        api = MemoryAPI(manager, auth_token=auth)
        assert api._auth_middleware is not None

    def test_memory_api_without_auth(self):
        """MemoryAPI 无认证时正常"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        api = MemoryAPI(manager)
        assert api._auth_middleware is None


# ═══════════════════════════════════════════════════════════════════════════════
# SC-7: 数据保留策略测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataRetentionPolicy:
    """SC-7: 数据保留策略"""

    async def test_retention_policy_config(self):
        """保留策略配置验证"""
        from cognitive_memory.core.memory_manager import DATA_RETENTION_POLICY

        assert DATA_RETENTION_POLICY[MemoryImportance.TRANSIENT] == 7
        assert DATA_RETENTION_POLICY[MemoryImportance.LOW] == 90
        assert DATA_RETENTION_POLICY[MemoryImportance.MEDIUM] == 365
        assert DATA_RETENTION_POLICY[MemoryImportance.HIGH] == -1
        assert DATA_RETENTION_POLICY[MemoryImportance.CRITICAL] == -1

    async def test_cleanup_by_importance(self):
        """按重要性清理过期数据"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")

        # 创建一个过期的 LOW 记忆 (100天前，LOW保留90天)
        old_time = datetime.now() - timedelta(days=100)
        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "test"},
            importance=MemoryImportance.LOW,
            created_at=old_time,
        )
        await manager._long_term.store(item)

        # 创建一个新的 LOW 记忆
        new_item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "new"},
            importance=MemoryImportance.LOW,
            created_at=datetime.now(),
        )
        await manager._long_term.store(new_item)

        # 创建一个 HIGH 记忆 (永久保留)
        high_item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "important"},
            importance=MemoryImportance.HIGH,
            created_at=old_time,
        )
        await manager._long_term.store(high_item)

        # 执行清理
        result = await manager._execute_retention_cleanup()
        assert "total_cleaned" in result
        assert "by_level" in result

        # 验证 HIGH 记忆未被清理
        retrieved = await manager._long_term.retrieve(high_item.id)
        assert retrieved is not None, "HIGH importance data should be retained"

    async def test_retention_logging(self):
        """数据清理日志记录"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        await manager._execute_retention_cleanup()
        log = manager.get_retention_log()
        assert len(log) >= 1
        assert "timestamp" in log[-1]

    async def test_backup_and_restore(self):
        """数据备份与恢复"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        old_time = datetime.now() - timedelta(days=10)

        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "backup_test"},
            importance=MemoryImportance.TRANSIENT,
            created_at=old_time,
        )
        await manager._long_term.store(item)

        # 手动备份
        manager._backup_data(item.id, item.to_dict())

        # 验证备份存在
        backup = manager.get_backup_data(item.id)
        assert backup is not None

    async def test_restore_data(self):
        """数据恢复"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")

        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "restore_test"},
            importance=MemoryImportance.LOW,
        )
        await manager._long_term.store(item)

        # 备份后删除
        manager._backup_data(item.id, item.to_dict())
        await manager._long_term.delete(item.id)

        # 恢复
        restored = await manager.restore_data(item.id)
        assert restored

        # 验证恢复
        retrieved = await manager._long_term.retrieve(item.id)
        assert retrieved is not None

    async def test_restore_nonexistent(self):
        """恢复不存在的数据"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        result = await manager.restore_data("nonexistent_id")
        assert not result

    async def test_get_retention_due_items(self):
        """获取即将清理的数据列表"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        old_time = datetime.now() - timedelta(days=100)

        # 使用一个唯一的 user_id 确保数据不被其他测试干扰
        import uuid
        uid = f"due_test_{uuid.uuid4().hex[:8]}"

        # LOW 级别在90天后过期，100天前创建的数据应该被清理
        item = MemoryItem(
            user_id=uid,
            memory_type=MemoryType.EPISODIC,
            content={"action": "due_test"},
            importance=MemoryImportance.LOW,
            created_at=old_time,
        )
        await manager._long_term.store(item)

        due_items = await manager.get_retention_due_items()
        found = any(d["data_id"] == item.id for d in due_items)
        assert found, f"Expected item {item.id} to be in due items, got {len(due_items)} items"

    async def test_high_importance_not_cleaned(self):
        """HIGH 级别数据不会被清理"""
        from cognitive_memory.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=":memory:")
        old_time = datetime.now() - timedelta(days=500)

        item = MemoryItem(
            user_id="user_001",
            memory_type=MemoryType.EPISODIC,
            content={"action": "permanent"},
            importance=MemoryImportance.HIGH,
            created_at=old_time,
        )
        await manager._long_term.store(item)

        await manager._execute_retention_cleanup()

        retrieved = await manager._long_term.retrieve(item.id)
        assert retrieved is not None, "HIGH importance should never be cleaned"