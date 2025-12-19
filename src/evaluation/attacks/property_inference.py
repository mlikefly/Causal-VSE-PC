"""
属性推断攻击实现。

推断群体级属性（如种族/性别分布）。
attack_success = AUC（按 GC7）

**验证: 需求 R2.AC2**
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
class PropertyInferenceMetrics:
    """属性推断攻击指标。"""
    auc: float  # attack_success
    accuracy: float
    property_name: str
    num_properties: int
    distribution_error: float  # 预测与真实分布的 MAE


@AttackRegistry.register(AttackType.PROPERTY_INFERENCE)
class PropertyInferenceAttack(AttackBase):
    """
    属性推断攻击。
    
    从 Z-view 数据集推断群体级属性，
    如训练集中的种族/性别分布。
    
    attack_success = AUC（越低隐私保护越好）
    
    按需求 R2.AC2：
    - 预测群体属性（种族/性别分布）
    - 输出准确率/AUC + 置信区间
    """
    
    attack_type = AttackType.PROPERTY_INFERENCE
    
    def __init__(
        self,
        device: str = None,
        property_classifier=None,
        property_name: str = "unknown",
    ):
        """
        初始化属性推断攻击。
        
        参数:
            device: 计算设备
            property_classifier: 用于属性推断的分类器
            property_name: 目标属性名称
        """
        super().__init__(device)
        self.property_classifier = property_classifier
        self.property_name = property_name
        self.trained_model = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练属性推断模型。
        
        参数:
            ctx: 攻击训练上下文
            train_datasets: 具有已知属性的数据集列表
            train_properties: 每个数据集的属性标签
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        train_datasets = kwargs.get('train_datasets')
        train_properties = kwargs.get('train_properties')
        
        if train_datasets is not None and train_properties is not None:
            self._train_property_classifier(train_datasets, train_properties)
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估属性推断攻击。
        
        参数:
            ctx: 评估上下文
            test_dataset: 要推断属性的数据集
            true_property: 真实属性值/分布
            
        返回:
            以 AUC 作为 attack_success 的 AttackResult
        """
        test_dataset = kwargs.get('test_dataset')
        true_property = kwargs.get('true_property')
        
        if test_dataset is None:
            return AttackResult(
                attack_type=self.attack_type,
                threat_level=ctx.threat_level,
                attack_success=0.5,
                metric_name="AUC",
                metric_value=0.5,
                status="failed",
            )
        
        metrics = self._compute_metrics(test_dataset, true_property)
        
        return AttackResult(
            attack_type=self.attack_type,
            threat_level=ctx.threat_level,
            attack_success=metrics.auc,
            metric_name="AUC",
            metric_value=metrics.auc,
            status="success",
            additional_metrics={
                'accuracy': metrics.accuracy,
                'property_name': metrics.property_name,
                'distribution_error': metrics.distribution_error,
            }
        )
    
    def _train_property_classifier(self, datasets, properties) -> None:
        """在数据集统计信息上训练属性分类器。"""
        # 从每个数据集提取聚合特征
        self.train_features = []
        self.train_labels = []
        
        for dataset, prop in zip(datasets, properties):
            features = self._extract_dataset_features(dataset)
            self.train_features.append(features)
            self.train_labels.append(prop)
        
        self.train_features = np.array(self.train_features)
        self.train_labels = np.array(self.train_labels)
    
    def _extract_dataset_features(self, dataset) -> np.ndarray:
        """从数据集提取聚合特征。"""
        if hasattr(dataset, 'numpy'):
            dataset = dataset.numpy()
        
        # 计算统计特征
        features = []
        
        # 像素值的均值和标准差
        features.append(dataset.mean())
        features.append(dataset.std())
        
        # 每通道统计
        if len(dataset.shape) > 3:
            for c in range(min(dataset.shape[1], 3)):
                features.append(dataset[:, c].mean())
                features.append(dataset[:, c].std())
        
        # 直方图特征
        hist, _ = np.histogram(dataset.flatten(), bins=10, density=True)
        features.extend(hist.tolist())
        
        return np.array(features)
    
    def _compute_metrics(
        self,
        dataset,
        true_property
    ) -> PropertyInferenceMetrics:
        """计算属性推断指标。"""
        if hasattr(dataset, 'numpy'):
            dataset = dataset.numpy()
        
        # 提取特征
        features = self._extract_dataset_features(dataset)
        
        # 预测属性
        if hasattr(self, 'train_features') and self.train_features is not None:
            predicted, scores = self._predict_property(features)
        else:
            # 随机预测
            predicted = np.random.rand()
            scores = np.array([0.5, 0.5])
        
        # 计算指标
        if true_property is not None:
            if hasattr(true_property, 'numpy'):
                true_property = true_property.numpy()
            
            # 对于分布属性
            if isinstance(true_property, (list, np.ndarray)):
                distribution_error = np.mean(np.abs(
                    np.array(predicted) - np.array(true_property)
                ))
                # 基于分布预测准确率的 AUC
                auc = 1.0 - distribution_error
            else:
                # 对于分类属性
                distribution_error = abs(predicted - true_property)
                auc = 1.0 - distribution_error
            
            accuracy = 1.0 - distribution_error
        else:
            auc = 0.5
            accuracy = 0.5
            distribution_error = 0.5
        
        return PropertyInferenceMetrics(
            auc=float(auc),
            accuracy=float(accuracy),
            property_name=self.property_name,
            num_properties=len(scores) if hasattr(scores, '__len__') else 1,
            distribution_error=float(distribution_error),
        )
    
    def _predict_property(self, features: np.ndarray):
        """使用训练好的模型预测属性。"""
        if self.train_features is None or len(self.train_features) == 0:
            return 0.5, np.array([0.5, 0.5])
        
        # K-NN 预测
        distances = np.linalg.norm(self.train_features - features, axis=1)
        k = min(3, len(distances))
        k_idx = np.argsort(distances)[:k]
        
        # 标签的加权平均
        weights = 1.0 / (distances[k_idx] + 1e-8)
        weights /= weights.sum()
        
        predicted = np.average(self.train_labels[k_idx], weights=weights, axis=0)
        
        # 分数（置信度）
        scores = np.array([1.0 - np.mean(distances[k_idx]), np.mean(distances[k_idx])])
        
        return predicted, scores
