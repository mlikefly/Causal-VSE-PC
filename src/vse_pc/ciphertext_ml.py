"""
密文域ML适配器

功能：
- 让标准ML模型能在加密图像上工作
- 支持分类、分割、检测任务
- 性能优化

设计思路：
- 加密保留了频域结构（低频语义信息）
- ML模型主要依赖低频，因此加密后仍能工作
- 无需修改ML模型，直接使用标准模型
"""

import torch
import torch.nn as nn
from typing import Dict, Union, Tuple, Optional


class CiphertextMLAdapter:
    """
    密文域ML适配器
    
    让标准ML模型能在加密图像上直接工作，无需解密。
    
    原理：
    - 加密保留了低频语义信息
    - ML模型（CNN）主要依赖低频特征
    - 因此加密后仍能工作，但性能略有下降
    """
    
    def __init__(self, base_model: nn.Module = None):
        """
        初始化适配器
        
        Args:
            base_model: 基础ML模型（可选，可在forward时传入）
        """
        self.base_model = base_model
    
    def forward(
        self,
        encrypted_image: torch.Tensor,
        model: Optional[nn.Module] = None,
        task_type: str = 'classification'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        在加密图像上运行ML模型
        
        Args:
            encrypted_image: [B, C, H, W] 加密图像，值域[0, 1]
            model: ML模型（如果初始化时未提供）
            task_type: 任务类型 {'classification', 'segmentation', 'detection'}
        
        Returns:
            predictions: 预测结果
                - classification: [B, num_classes] logits
                - segmentation: [B, num_classes, H, W] masks
                - detection: (boxes, scores, labels)
        """
        # 使用传入的模型或初始化时的模型
        ml_model = model if model is not None else self.base_model
        if ml_model is None:
            raise ValueError("必须提供ML模型（通过参数或初始化）")
        
        # 关键：直接在加密图像上运行，无需解密
        # 加密保留了频域结构，ML模型仍能工作
        
        ml_model.eval()  # 确保是评估模式
        with torch.no_grad():
            if task_type == 'classification':
                # 分类任务：直接前向传播
                logits = ml_model(encrypted_image)
                return logits
                
            elif task_type == 'segmentation':
                # 分割任务：直接前向传播
                masks = ml_model(encrypted_image)
                return masks
                
            elif task_type == 'detection':
                # 检测任务：可能需要后处理
                outputs = ml_model(encrypted_image)
                # 假设输出格式：(boxes, scores, labels)
                return outputs
                
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
    
    def evaluate_performance(
        self,
        encrypted_images: torch.Tensor,
        ground_truth: torch.Tensor,
        model: nn.Module,
        task_type: str = 'classification'
    ) -> Dict[str, float]:
        """
        评估密文域ML性能
        
        Args:
            encrypted_images: [B, C, H, W] 加密图像
            ground_truth: 真实标签/掩码
            model: ML模型
            task_type: 任务类型
        
        Returns:
            metrics: 性能指标字典
        """
        predictions = self.forward(encrypted_images, model, task_type)
        
        if task_type == 'classification':
            # 计算准确率
            pred_labels = torch.argmax(predictions, dim=1)
            accuracy = (pred_labels == ground_truth).float().mean().item()
            return {'accuracy': accuracy}
            
        elif task_type == 'segmentation':
            # 计算mIoU
            pred_masks = torch.argmax(predictions, dim=1)
            # 简化：使用像素准确率
            pixel_acc = (pred_masks == ground_truth).float().mean().item()
            return {'pixel_accuracy': pixel_acc}
            
        elif task_type == 'detection':
            # 计算mAP（简化版）
            # 这里需要根据实际检测模型输出格式计算
            return {'mAP': 0.0}  # 占位符
        
        return {}


class CiphertextClassificationAdapter(CiphertextMLAdapter):
    """分类任务专用适配器"""
    
    def __init__(self, classifier: nn.Module):
        super().__init__(classifier)
        self.classifier = classifier
    
    def classify(self, encrypted_image: torch.Tensor) -> torch.Tensor:
        """分类加密图像"""
        return self.forward(encrypted_image, self.classifier, 'classification')


class CiphertextSegmentationAdapter(CiphertextMLAdapter):
    """分割任务专用适配器"""
    
    def __init__(self, segmenter: nn.Module):
        super().__init__(segmenter)
        self.segmenter = segmenter
    
    def segment(self, encrypted_image: torch.Tensor) -> torch.Tensor:
        """分割加密图像"""
        return self.forward(encrypted_image, self.segmenter, 'segmentation')


def test_ciphertext_ml():
    """测试密文域ML适配器"""
    print("测试密文域ML适配器")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模拟模型
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 10, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 创建适配器和模型
    classifier = SimpleClassifier().to(device)
    adapter = CiphertextClassificationAdapter(classifier)
    
    # 创建测试加密图像
    encrypted_image = torch.rand(2, 1, 256, 256, device=device)
    
    # 测试分类
    predictions = adapter.classify(encrypted_image)
    print(f"  分类输出形状: {predictions.shape}")
    print(f"  输出范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    print("✓ 测试完成")


if __name__ == "__main__":
    test_ciphertext_ml()

