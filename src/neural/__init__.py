"""
神经网络模块
包含U-Net等神经网络组件

注意：GNN策略网络相关组件已移除，当前Causal-VSE-PC项目使用因果推断驱动的策略。
"""

from .unet import UNetSaliencyDetector

__all__ = [
    'UNetSaliencyDetector',
]
