"""
SCNE 主加密系统模块

包含:
- SCNECipher: 核心加密器
- SCNECipherAPI: 简化 API
- DualViewEncryptionEngine: 双视图加密引擎
- EncryptionResult: 加密结果数据类
"""

from .scne_cipher import SCNECipher, SCNECipherAPI
from .dual_view_engine import DualViewEncryptionEngine, EncryptionResult

__all__ = [
    'SCNECipher',
    'SCNECipherAPI',
    'DualViewEncryptionEngine',
    'EncryptionResult'
]
