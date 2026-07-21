"""
认知记忆模块 - 数据加密与隐私保护

提供用户数据的加密存储和隐私保护机制:
- AES-256-GCM 加密敏感数据字段 (SC-1: 强制使用 cryptography 库)
- 数据脱敏（用于日志和调试输出）(SC-5: 增强递归脱敏)
- 用户数据匿名化
- 数据访问审计 (SC-4: 集中化审计日志)
- 数据分级加密存储 (SC-3: 差异化存储策略)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from base64 import b64encode, b64decode
from datetime import datetime
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..models.memory import MemoryImportance

logger = logging.getLogger(__name__)


class MemoryEncryption:
    """记忆数据加密器 - AES-256-GCM 加密 (SC-1: 无降级方案)"""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or self._generate_key()

    @staticmethod
    def _generate_key() -> bytes:
        """生成加密密钥"""
        return os.urandom(32)

    def encrypt(self, data: dict[str, Any]) -> dict[str, str]:
        """加密敏感数据字段

        使用 AES-256-GCM 模式加密。

        Returns:
            {"ciphertext": "...", "nonce": "..."}
        """
        aesgcm = AESGCM(self._key)
        nonce = os.urandom(12)
        plaintext = json.dumps(data, ensure_ascii=False).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        return {
            "ciphertext": b64encode(ciphertext).decode("ascii"),
            "nonce": b64encode(nonce).decode("ascii"),
        }

    def decrypt(self, encrypted: dict[str, str]) -> dict[str, Any]:
        """解密数据"""
        aesgcm = AESGCM(self._key)
        nonce = b64decode(encrypted["nonce"])
        ciphertext = b64decode(encrypted["ciphertext"])
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))


class PrivacyManager:
    """隐私保护管理器 (SC-5: 增强脱敏)"""

    # 敏感字段列表 (SC-5: 可配置化管理)
    SENSITIVE_FIELDS = {
        "name",
        "phone_number",
        "email",
        "home_address",
        "work_address",
        "health_conditions",
        "exact_location",
        "id_card",
        "bank_account",
        "license_plate",
        "vin",
    }

    # SC-5: 差异化脱敏策略配置
    FIELD_MASK_STRATEGIES = {
        "phone_number": "phone",
        "email": "email",
        "id_card": "id_card",
        "name": "name",
        "home_address": "address",
        "work_address": "address",
        "health_conditions": "full",
        "exact_location": "address",
        "bank_account": "bank_account",
        "license_plate": "license_plate",
        "vin": "partial",
    }

    @classmethod
    def add_sensitive_field(cls, field_name: str, strategy: str = "default"):
        """SC-5: 动态添加敏感字段"""
        cls.SENSITIVE_FIELDS.add(field_name)
        cls.FIELD_MASK_STRATEGIES[field_name] = strategy

    @classmethod
    def remove_sensitive_field(cls, field_name: str):
        """SC-5: 动态移除敏感字段"""
        cls.SENSITIVE_FIELDS.discard(field_name)
        cls.FIELD_MASK_STRATEGIES.pop(field_name, None)

    @classmethod
    def mask_sensitive_data(cls, data: dict[str, Any], depth: int = 0, max_depth: int = 10) -> dict[str, Any]:
        """SC-5: 递归深度遍历脱敏处理

        对敏感字段进行掩码，支持递归处理嵌套结构。

        Args:
            data: 待脱敏数据
            depth: 当前递归深度
            max_depth: 最大递归深度

        Returns:
            脱敏后的数据
        """
        if depth > max_depth:
            return data

        masked = {}
        for key, value in data.items():
            if key in cls.SENSITIVE_FIELDS:
                strategy = cls.FIELD_MASK_STRATEGIES.get(key, "default")
                masked[key] = cls._apply_mask_strategy(value, strategy)
            elif isinstance(value, dict):
                masked[key] = cls.mask_sensitive_data(value, depth + 1, max_depth)
            elif isinstance(value, list):
                masked[key] = [
                    cls.mask_sensitive_data(v, depth + 1, max_depth) if isinstance(v, dict)
                    else (cls._apply_mask_strategy(v, "default") if isinstance(v, str) and cls._is_sensitive_value(v) else v)
                    for v in value
                ]
            else:
                masked[key] = value
        return masked

    @staticmethod
    def _is_sensitive_value(value: str) -> bool:
        """SC-5: 检测值是否可能为敏感数据（基于模式匹配）"""
        if not isinstance(value, str):
            return False
        # 检测手机号模式 (11位数字)
        if len(value) == 11 and value.isdigit() and value.startswith("1"):
            return True
        # 检测邮箱模式
        if "@" in value and "." in value.split("@")[-1]:
            return True
        return False

    @staticmethod
    def _apply_mask_strategy(value: Any, strategy: str) -> Any:
        """SC-5: 根据策略应用差异化脱敏

        Strategies:
            - phone: 138****1234
            - email: j***@example.com
            - id_card: 110***********1234
            - name: 张*
            - address: 保留前3字符
            - bank_account: ****1234
            - license_plate: 京A***8
            - full: 完全掩码 ***
            - partial: 保留首尾各1个字符
            - default: 保留首尾各1个字符
        """
        if not isinstance(value, str):
            return "***"

        if strategy == "phone":
            if len(value) < 7:
                return "***"
            return value[:3] + "****" + value[-4:]
        elif strategy == "email":
            if "@" in value:
                local, domain = value.split("@", 1)
                if len(local) <= 2:
                    return "*@" + domain
                return local[0] + "***@" + domain
            return "***@***"
        elif strategy == "id_card":
            if len(value) >= 6:
                return value[:3] + "***********" + value[-4:]
            return value[0] + "*" * (len(value) - 2) + value[-1] if len(value) > 2 else "**"
        elif strategy == "name":
            if len(value) <= 1:
                return "*"
            return value[0] + "*" * (len(value) - 1)
        elif strategy == "address":
            return value[:3] + "***" if len(value) > 3 else "***"
        elif strategy == "bank_account":
            return "****" + value[-4:] if len(value) >= 4 else "****"
        elif strategy == "license_plate":
            if len(value) >= 3:
                return value[:1] + "***" + value[-1:]
            return "***"
        elif strategy == "full":
            return "***"
        else:
            # default/partial
            if len(value) <= 2:
                return "**"
            return value[0] + "*" * (len(value) - 2) + value[-1]

    @staticmethod
    def anonymize_user_id(user_id: str) -> str:
        """匿名化用户ID"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    @staticmethod
    def create_data_hash(data: dict[str, Any]) -> str:
        """创建数据完整性校验哈希"""
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def verify_data_integrity(data: dict[str, Any], expected_hash: str) -> bool:
        """验证数据完整性"""
        return PrivacyManager.create_data_hash(data) == expected_hash


# ═══════════════════════════════════════════════════════════════════════════════
# SC-3: 数据分级加密存储
# ═══════════════════════════════════════════════════════════════════════════════


class TieredEncryption:
    """SC-3: 基于 MemoryImportance 的差异化加密存储策略

    分级策略:
    - CRITICAL/HIGH: 端到端加密存储，密钥安全管理
    - MEDIUM/LOW: 明文存储，确保传输安全
    - TRANSIENT: 不持久化，仅内存存储
    """

    def __init__(self, encryption_key: Optional[bytes] = None):
        self._encryption = MemoryEncryption(key=encryption_key)

    def should_encrypt(self, importance: MemoryImportance) -> bool:
        """判断是否需要加密存储"""
        return importance in (MemoryImportance.CRITICAL, MemoryImportance.HIGH)

    def should_persist(self, importance: MemoryImportance) -> bool:
        """判断是否需要持久化存储"""
        return importance != MemoryImportance.TRANSIENT

    def encrypt_if_needed(self, data: dict[str, Any], importance: MemoryImportance) -> dict[str, Any]:
        """根据重要性加密数据"""
        if self.should_encrypt(importance):
            encrypted = self._encryption.encrypt(data)
            return {
                "_encrypted": True,
                "_importance": importance.value,
                "ciphertext": encrypted["ciphertext"],
                "nonce": encrypted["nonce"],
            }
        return {
            "_encrypted": False,
            "_importance": importance.value,
            **data,
        }

    def decrypt_if_needed(self, stored_data: dict[str, Any]) -> dict[str, Any]:
        """根据存储标记解密数据"""
        if stored_data.get("_encrypted", False):
            encrypted = {
                "ciphertext": stored_data["ciphertext"],
                "nonce": stored_data["nonce"],
            }
            return self._encryption.decrypt(encrypted)
        return {k: v for k, v in stored_data.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════════════
# SC-4: 审计日志集中化
# ═══════════════════════════════════════════════════════════════════════════════


class AuditLogger:
    """SC-4: 数据访问审计日志 - 集中化管理

    支持输出到:
    - 本地文件 (JSON 格式)
    - 远程日志平台 (ELK/Splunk)

    日志格式: 结构化 JSON，包含必要字段。
    支持重试机制和完整性校验。
    """

    # 日志平台类型
    PLATFORM_LOCAL = "local"
    PLATFORM_ELK = "elk"
    PLATFORM_SPLUNK = "splunk"

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0

    def __init__(
        self,
        log_path: str = "audit.log",
        platform: str = "local",
        platform_config: Optional[dict[str, Any]] = None,
        service_name: str = "cognitive-memory",
    ):
        self._log_path = log_path
        self._platform = platform
        self._platform_config = platform_config or {}
        self._service_name = service_name
        self._hmac_key = os.urandom(32)

        self._init_local_logger()

    def _init_local_logger(self):
        """初始化本地日志记录器"""
        self._logger = logging.getLogger("cognitive_memory.audit")
        self._logger.setLevel(logging.INFO)
        self._handler = logging.FileHandler(self._log_path)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)

    def close(self):
        """关闭审计日志资源"""
        if hasattr(self, '_handler') and self._handler:
            self._handler.close()
            self._logger.removeHandler(self._handler)
            self._handler = None

    def _build_audit_entry(
        self,
        user_id: str,
        operation: str,
        data_type: str,
        requester: str = "system",
        success: bool = True,
        ip_address: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """SC-4: 构建结构化 JSON 审计日志条目"""
        anonymized_id = PrivacyManager.anonymize_user_id(user_id)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self._service_name,
            "operator": anonymized_id,
            "operation_type": operation,
            "target": data_type,
            "requester": requester,
            "result": "success" if success else "failure",
            "ip_address": ip_address,
            "details": details or {},
        }
        # 完整性校验
        entry["integrity_hash"] = self._compute_integrity_hash(entry)
        return entry

    def _compute_integrity_hash(self, entry: dict[str, Any]) -> str:
        """SC-4: 计算日志完整性哈希 (不可篡改)"""
        content = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        return hmac.new(
            self._hmac_key, content.encode(), hashlib.sha256
        ).hexdigest()

    def verify_log_integrity(self, entry: dict[str, Any]) -> bool:
        """SC-4: 验证日志完整性"""
        if "integrity_hash" not in entry:
            return False
        expected_hash = entry.pop("integrity_hash")
        return self._compute_integrity_hash(entry) == expected_hash

    def log_access(
        self,
        user_id: str,
        operation: str,
        data_type: str,
        requester: str = "system",
        success: bool = True,
        ip_address: str = "",
        details: Optional[dict[str, Any]] = None,
    ):
        """SC-4: 记录数据访问日志 (支持重试)"""
        entry = self._build_audit_entry(
            user_id=user_id,
            operation=operation,
            data_type=data_type,
            requester=requester,
            success=success,
            ip_address=ip_address,
            details=details,
        )

        # 本地日志写入
        self._write_local(entry)

        # 远程日志平台发送
        if self._platform != self.PLATFORM_LOCAL:
            self._send_remote(entry)

    def _write_local(self, entry: dict[str, Any]):
        """写入本地日志文件"""
        try:
            self._logger.info(json.dumps(entry, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Audit local write failed: {e}")

    def _send_remote(self, entry: dict[str, Any]):
        """SC-4: 发送日志到远程平台 (带重试机制)"""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self._platform == self.PLATFORM_ELK:
                    self._send_to_elk(entry)
                elif self._platform == self.PLATFORM_SPLUNK:
                    self._send_to_splunk(entry)
                break
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_S * (attempt + 1))
                    logger.warning(f"Audit remote send retry {attempt + 1}/{self.MAX_RETRIES}: {e}")
                else:
                    logger.error(f"Audit remote send failed after {self.MAX_RETRIES} retries: {e}")
                    # 最终失败时写入本地作为兜底
                    self._write_local({**entry, "_remote_failed": True})

    def _send_to_elk(self, entry: dict[str, Any]):
        """发送到 ELK (Elasticsearch) 日志平台"""
        endpoint = self._platform_config.get("elk_endpoint", "")
        if not endpoint:
            return

        import urllib.request

        data = json.dumps(entry, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/audit-log/_doc",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)

    def _send_to_splunk(self, entry: dict[str, Any]):
        """发送到 Splunk 日志平台"""
        endpoint = self._platform_config.get("splunk_endpoint", "")
        token = self._platform_config.get("splunk_token", "")
        if not endpoint or not token:
            return

        import urllib.request

        data = json.dumps({"event": entry}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/services/collector",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Splunk {token}",
            },
        )
        urllib.request.urlopen(req, timeout=5)