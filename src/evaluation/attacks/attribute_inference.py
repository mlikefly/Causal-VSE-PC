"""
属性推断攻击实现。

attack_success = AUC（按 GC7）

**验证: 需求 R2.AC4**
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..attack_framework import (
    AttackBase,
    AttackFitContext,
    AttackEvalContext,
    AttackResult,
    AttackType,
    AttackRegistry,
)


@dataclass
class AttributeInferenceMetrics:
    """属性推断攻击指标。"""
    auc: float  # attack_success
    accuracy: float
    attribute_name: str
    num_classes: int
    num_samples: int


@AttackRegistry.register(AttackType.ATTRIBUTE_INFERENCE)
class AttributeInferenceAttack(AttackBase):
    """
    属性推断攻击。
    
    评估攻击者是否能从 Z-view 推断敏感属性
    （如种族、性别、年龄）。
    
    attack_success = AUC（越低隐私保护越好）
    """
    
    attack_type = AttackType.ATTRIBUTE_INFERENCE
    
    def __init__(self, device: str = None, classifier=None, attribute_name: str = "unknown"):
        """
        初始化属性推断攻击。
        
        参数:
            device: 计算设备
            classifier: 属性分类器模型
            attribute_name: 目标属性名称
        """
        super().__init__(device)
        self.classifier = classifier
        self.attribute_name = attribute_name
        self.trained_classifier = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练属性分类器。
        
        参数:
            ctx: 攻击训练上下文
            train_images: 训练图像（Z-view）
            train_labels: 属性标签
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        train_images = kwargs.get('train_images')
        train_labels = kwargs.get('train_labels')
        
        if train_images is not None and train_labels is not None:
            self._train_classifier(train_images, train_labels)
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估属性推断攻击。
        
        参数:
            ctx: 评估上下文
            test_images: 测试图像（Z-view）
            test_labels: 真实属性标签
            
        返回:
            以 AUC 作为 attack_success 的 AttackResult
        """
        test_images = kwargs.get('test_images')
        test_labels = kwargs.get('test_labels')
        
        if test_images is None or test_labels is None:
            return AttackResult(
                attack_type=self.attack_type,
                threat_level=ctx.threat_level,
                attack_success=0.5,  # Random guess
                metric_name="AUC",
                metric_value=0.5,
                status="failed",
            )
        
        metrics = self._compute_metrics(test_images, test_labels)
        
        return AttackResult(
            attack_type=self.attack_type,
            threat_level=ctx.threat_level,
            attack_success=metrics.auc,
            metric_name="AUC",
            metric_value=metrics.auc,
            status="success",
            additional_metrics={
                'accuracy': metrics.accuracy,
                'attribute_name': metrics.attribute_name,
                'num_classes': metrics.num_classes,
            }
        )
    
    def _train_classifier(self, images, labels) -> None:
        """训练简单分类器。"""
        # 提取特征
        features = self._extract_features(images)
        
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        
        # 存储用于 k-NN 分类
        self.train_features = features
        self.train_labels = labels
    
    def _extract_features(self, images) -> np.ndarray:
        """从图像中提取特征。"""
        if self.classifier is not None:
            import torch
            with torch.no_grad():
                if hasattr(images, 'to'):
                    images = images.to(self.device)
                features = self.classifier.extract_features(images)
                if hasattr(features, 'cpu'):
                    features = features.cpu().numpy()
            return features
        
        # 回退：展平图像
        if hasattr(images, 'numpy'):
            images = images.numpy()
        return images.reshape(len(images), -1)
    
    def _compute_metrics(self, images, labels) -> AttributeInferenceMetrics:
        """计算属性推断指标。"""
        features = self._extract_features(images)
        
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        
        # 使用 k-NN 或分类器预测
        if hasattr(self, 'train_features') and self.train_features is not None:
            predictions, scores = self._knn_predict(features)
        else:
            # 随机预测
            num_classes = len(np.unique(labels))
            predictions = np.random.randint(0, num_classes, len(labels))
            scores = np.random.rand(len(labels), num_classes)
        
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        
        # 计算 AUC（多分类使用 one-vs-rest）
        auc = self._compute_multiclass_auc(scores, labels)
        
        return AttributeInferenceMetrics(
            auc=auc,
            accuracy=accuracy,
            attribute_name=self.attribute_name,
            num_classes=len(np.unique(labels)),
            num_samples=len(labels),
        )
    
    def _knn_predict(self, features: np.ndarray, k: int = 5):
        """K-NN 预测。"""
        predictions = []
        scores = []
        
        num_classes = len(np.unique(self.train_labels))
        
        for feat in features:
            # 计算距离
            dists = np.linalg.norm(self.train_features - feat, axis=1)
            
            # 获取 k 个最近邻
            k_idx = np.argsort(dists)[:k]
            k_labels = self.train_labels[k_idx]
            
            # 投票
            pred = np.bincount(k_labels.astype(int), minlength=num_classes).argmax()
            predictions.append(pred)
            
            # 分数（逆距离加权）
            score = np.zeros(num_classes)
            for idx, label in zip(k_idx, k_labels):
                score[int(label)] += 1.0 / (dists[idx] + 1e-8)
            score /= score.sum() + 1e-8
            scores.append(score)
        
        return np.array(predictions), np.array(scores)
    
    def _compute_multiclass_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """计算多分类的宏平均 AUC。"""
        num_classes = scores.shape[1] if len(scores.shape) > 1 else len(np.unique(labels))
        
        aucs = []
        for c in range(num_classes):
            binary_labels = (labels == c).astype(int)
            if len(scores.shape) > 1:
                class_scores = scores[:, c]
            else:
                class_scores = (scores == c).astype(float)
            
            auc = self._compute_binary_auc(class_scores, binary_labels)
            aucs.append(auc)
        
        return float(np.mean(aucs))
    
    def _compute_binary_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """计算二分类 AUC。"""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        correct = 0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    correct += 1
                elif ps == ns:
                    correct += 0.5
        
        return correct / (len(pos_scores) * len(neg_scores))
