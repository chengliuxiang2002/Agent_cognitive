"""
认知记忆模块 - 数据加密与隐私保护

提供用户数据的加密存储和隐私保护机制:
- AES-256-GCM 加密敏感数据字段
- 数据脱敏（用于日志和调试输出）
- 用户数据匿名化
- 数据访问审计
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from base64 import b64encode, b64decode
from typing import Any, Optional


class MemoryEncryption:
    """记忆数据加密器 - AES-256-GCM 加密"""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or self._generate_key()

    @staticmethod
    def _generate_key() -> bytes:
        """生成加密密钥"""
        return os.urandom(32)

    def encrypt(self, data: dict[str, Any]) -> dict[str, str]:
        """加密敏感数据字段

        使用 AES-256-GCM 模式加密。
        注意: 生产环境应使用 cryptography 库，此处为简化实现。

        Returns:
            {"ciphertext": "...", "nonce": "...", "tag": "..."}
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            aesgcm = AESGCM(self._key)
            nonce = os.urandom(12)
            plaintext = json.dumps(data, ensure_ascii=False).encode("utf-8")
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)

            return {
                "ciphertext": b64encode(ciphertext).decode("ascii"),
                "nonce": b64encode(nonce).decode("ascii"),
            }
        except ImportError:
            # 降级方案：使用简单的混淆（不推荐用于生产环境）
            return self._simple_encrypt(data)

    def decrypt(self, encrypted: dict[str, str]) -> dict[str, Any]:
        """解密数据"""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            aesgcm = AESGCM(self._key)
            nonce = b64decode(encrypted["nonce"])
            ciphertext = b64decode(encrypted["ciphertext"])
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return json.loads(plaintext.decode("utf-8"))
        except ImportError:
            return self._simple_decrypt(encrypted)

    def _simple_encrypt(self, data: dict[str, Any]) -> dict[str, str]:
        """简单加密（仅用于开发环境，不推荐生产使用）"""
        plaintext = json.dumps(data, ensure_ascii=False)
        # XOR 混淆
        key_str = self._key.hex()
        encrypted = "".join(
            chr(ord(c) ^ ord(key_str[i % len(key_str)]))
            for i, c in enumerate(plaintext)
        )
        return {"ciphertext": b64encode(encrypted.encode()).decode("ascii")}

    def _simple_decrypt(self, encrypted: dict[str, str]) -> dict[str, Any]:
        encrypted_str = b64decode(encrypted["ciphertext"]).decode()
        key_str = self._key.hex()
        plaintext = "".join(
            chr(ord(c) ^ ord(key_str[i % len(key_str)]))
            for i, c in enumerate(encrypted_str)
        )
        return json.loads(plaintext)


class PrivacyManager:
    """隐私保护管理器"""

    # 敏感字段列表
    SENSITIVE_FIELDS = {
        "name",
        "phone_number",
        "email",
        "home_address",
        "work_address",
        "health_conditions",
        "exact_location",
    }

    @staticmethod
    def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
        """脱敏处理：对敏感字段进行掩码"""
        masked = {}
        for key, value in data.items():
            if key in PrivacyManager.SENSITIVE_FIELDS:
                if isinstance(value, str):
                    if len(value) <= 2:
                        masked[key] = "**"
                    else:
                        masked[key] = value[0] + "*" * (len(value) - 2) + value[-1]
                elif isinstance(value, list):
                    masked[key] = [PrivacyManager.mask_sensitive_data({"v": v})["v"] if isinstance(v, dict) else "***" for v in value]
                else:
                    masked[key] = "***"
            elif isinstance(value, dict):
                masked[key] = PrivacyManager.mask_sensitive_data(value)
            else:
                masked[key] = value
        return masked

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


class AuditLogger:
    """数据访问审计日志"""

    def __init__(self, log_path: str = "audit.log"):
        self._log_path = log_path

    def log_access(
        self,
        user_id: str,
        operation: str,
        data_type: str,
        requester: str = "system",
        success: bool = True,
    ):
        """记录数据访问日志"""
        import logging

        audit_logger = logging.getLogger("cognitive_memory.audit")
        audit_logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self._log_path)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        audit_logger.addHandler(handler)

        anonymized_id = PrivacyManager.anonymize_user_id(user_id)
        audit_logger.info(
            f"ACCESS | user={anonymized_id} | op={operation} | "
            f"type={data_type} | by={requester} | success={success}"
        )