# -*- coding: utf-8 -*-
"""
SCNE 模型目录
重构后，此目录已清空
所有组件已移动到对应模块：
- cipher/: 主加密系统（SCNECipher）
- core/: 核心加密组件
- neural/: 神经网络组件
- crypto/: 密码学组件

为了向后兼容，提供重定向导入
"""

# 向后兼容：从新位置导入
from ..cipher import SCNECipher, SCNECipherAPI

__all__ = [
    'SCNECipher',
    'SCNECipherAPI'
]

# 提示用户使用新的导入路径
import warnings
warnings.warn(
    "从 src.models 导入 SCNECipher 已废弃，"
    "请使用 from src.cipher import SCNECipher",
    DeprecationWarning,
    stacklevel=2
)
