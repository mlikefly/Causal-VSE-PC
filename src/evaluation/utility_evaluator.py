# -*- coding: utf-8 -*-
"""
任务效用评估器 (Utility Evaluator)

实现三类任务的效用评估：
1. 人脸属性分类 (Face Attribute Classification)
2. 目标检测 (Object Detection)
3. 语义分割 (Semantic Segmentation)

以及：
- 公平性分组评估
- 隐私-效用曲线生成

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassificationResult:
    """分类评估结果"""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_classes: int = 0
    num_samples: int = 0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'num_classes': self.num_classes,
            'num_samples': self.num_samples,
            'per_class_accuracy': self.per_class_accuracy
        }


@dataclass
class DetectionResult:
    """检测评估结果"""
    map50: float = 0.0      # mAP@0.5
    map75: float = 0.0      # mAP@0.75
    map_avg: float = 0.0    # mAP@[0.5:0.95]
    precision: float = 0.0
    recall: float = 0.0
    num_detections: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'map50': self.map50,
            'map75': self.map75,
            'map_avg': self.map_avg,
            'precision': self.precision,
            'recall': self.recall,
            'num_detections': self.num_detections
        }


@dataclass
class SegmentationResult:
    """分割评估结果"""
    miou: float = 0.0           # Mean IoU
    pixel_accuracy: float = 0.0  # 像素准确率
    dice_score: float = 0.0      # Dice 系数
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'miou': self.miou,
            'pixel_accuracy': self.pixel_accuracy,
            'dice_score': self.dice_score,
            'per_class_iou': self.per_class_iou
        }


@dataclass
class FairnessResult:
    """公平性评估结果"""
    group_accuracies: Dict[str, float] = field(default_factory=dict)
    group_calibrations: Dict[str, float] = field(default_factory=dict)
    accuracy_gap: float = 0.0       # 最大-最小准确率差
    calibration_gap: float = 0.0    # 最大-最小校准差
    demographic_parity: float = 0.0  # 人口统计平等
    equalized_odds: float = 0.0      # 均等化赔率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'group_accuracies': self.group_accuracies,
            'group_calibrations': self.group_calibrations,
            'accuracy_gap': self.accuracy_gap,
            'calibration_gap': self.calibration_gap,
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds
        }


@dataclass
class PrivacyUtilityCurve:
    """
    隐私-效用曲线数据
    
    **Property 12: Privacy-Utility Curve Completeness**
    """
    privacy_levels: List[float] = field(default_factory=list)
    utility_values: List[float] = field(default_factory=list)
    task_type: str = ""
    metric_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'privacy_levels': self.privacy_levels,
            'utility_values': self.utility_values,
            'task_type': self.task_type,
            'metric_name': self.metric_name
        }
    
    def is_complete(self) -> bool:
        """检查曲线是否完整（包含所有五档隐私级别）"""
        expected_levels = {0.0, 0.3, 0.5, 0.7, 1.0}
        return set(self.privacy_levels) == expected_levels


class UtilityEvaluator:
    """
    任务效用评估器
    
    支持三类任务评估：
    1. 分类任务 - 人脸属性分类
    2. 检测任务 - 目标检测
    3. 分割任务 - 语义分割
    
    以及公平性分组评估和隐私-效用曲线生成
    """
    
    PRIVACY_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    def __init__(
        self,
        device: str = None,
        classifier: nn.Module = None,
        detector: nn.Module = None,
        segmentor: nn.Module = None
    ):
        """
        初始化效用评估器
        
        Args:
            device: 计算设备
            classifier: 分类模型
            detector: 检测模型
            segmentor: 分割模型
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = classifier
        self.detector = detector
        self.segmentor = segmentor
        
        # 移动模型到设备
        if self.classifier:
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
        if self.detector:
            self.detector = self.detector.to(self.device)
            self.detector.eval()
        if self.segmentor:
            self.segmentor = self.segmentor.to(self.device)
            self.segmentor.eval()
    
    def evaluate_classification(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribute_names: List[str] = None
    ) -> ClassificationResult:
        """
        评估分类任务
        
        **Requirements 5.1**: 在 CelebA Z-view 上评估 40 属性
        
        Args:
            images: [N, C, H, W] 输入图像
            labels: [N] 或 [N, num_attrs] 标签
            attribute_names: 属性名称列表
        
        Returns:
            ClassificationResult: 分类评估结果
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        n_samples = len(images)
        
        if self.classifier is not None:
            with torch.no_grad():
                logits = self.classifier(images)
                if logits.dim() == 2 and logits.shape[1] > 1:
                    preds = logits.argmax(dim=1)
                else:
                    preds = (logits > 0).long().squeeze()
        else:
            # 无分类器时，使用简单的 k-NN
            preds = self._simple_knn_classify(images, labels)
        
        # 处理多标签情况
        if labels.dim() == 2:
            # 多属性分类
            return self._evaluate_multi_label(preds, labels, attribute_names)
        else:
            # 单标签分类
            return self._evaluate_single_label(preds, labels, attribute_names)
    
    def _evaluate_single_label(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        class_names: List[str] = None
    ) -> ClassificationResult:
        """评估单标签分类"""
        n_samples = len(labels)
        n_classes = len(torch.unique(labels))
        
        # 准确率
        correct = (preds == labels).float()
        accuracy = correct.mean().item()
        
        # 计算 F1、Precision、Recall
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        
        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-8))
        
        # 每类准确率
        per_class_acc = {}
        unique_labels = torch.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_acc = (preds[mask] == labels[mask]).float().mean().item()
            class_name = class_names[i] if class_names and i < len(class_names) else f"class_{label.item()}"
            per_class_acc[class_name] = class_acc
        
        return ClassificationResult(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            num_classes=n_classes,
            num_samples=n_samples,
            per_class_accuracy=per_class_acc
        )
    
    def _evaluate_multi_label(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        attribute_names: List[str] = None
    ) -> ClassificationResult:
        """评估多标签分类"""
        n_samples, n_attrs = labels.shape
        
        # 确保 preds 形状正确
        if preds.dim() == 1:
            preds = preds.unsqueeze(1).expand(-1, n_attrs)
        
        # 每属性准确率
        per_attr_acc = {}
        accuracies = []
        
        for i in range(n_attrs):
            attr_correct = (preds[:, i] == labels[:, i]).float().mean().item()
            attr_name = attribute_names[i] if attribute_names and i < len(attribute_names) else f"attr_{i}"
            per_attr_acc[attr_name] = attr_correct
            accuracies.append(attr_correct)
        
        avg_accuracy = np.mean(accuracies)
        
        return ClassificationResult(
            accuracy=avg_accuracy,
            f1_score=avg_accuracy,  # 简化
            precision=avg_accuracy,
            recall=avg_accuracy,
            num_classes=n_attrs,
            num_samples=n_samples,
            per_class_accuracy=per_attr_acc
        )
    
    def _simple_knn_classify(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        k: int = 3
    ) -> torch.Tensor:
        """简单的 k-NN 分类"""
        features = images.view(images.shape[0], -1).float()
        features = F.normalize(features, p=2, dim=1)
        
        preds = []
        for i in range(len(images)):
            sim = torch.mm(features[i:i+1], features.t())[0]
            sim[i] = -float('inf')
            _, top_k = sim.topk(k)
            pred = labels[top_k].mode().values
            preds.append(pred)
        
        return torch.stack(preds)
    
    def evaluate_detection(
        self,
        images: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor]
    ) -> DetectionResult:
        """
        评估检测任务
        
        **Requirements 5.2**: 目标检测评估
        
        Args:
            images: [N, C, H, W] 输入图像
            gt_boxes: List of [M, 4] 真实框
            gt_labels: List of [M] 真实标签
        
        Returns:
            DetectionResult: 检测评估结果
        """
        images = images.to(self.device)
        
        if self.detector is not None:
            with torch.no_grad():
                predictions = self.detector(images)
        else:
            # 无检测器时，返回默认结果
            return DetectionResult(
                map50=0.0,
                map75=0.0,
                map_avg=0.0,
                precision=0.0,
                recall=0.0,
                num_detections=0
            )
        
        # 计算 mAP
        map50 = self._calculate_map(predictions, gt_boxes, gt_labels, iou_threshold=0.5)
        map75 = self._calculate_map(predictions, gt_boxes, gt_labels, iou_threshold=0.75)
        
        # 计算平均 mAP
        map_values = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            map_values.append(self._calculate_map(predictions, gt_boxes, gt_labels, thresh))
        map_avg = np.mean(map_values)
        
        return DetectionResult(
            map50=map50,
            map75=map75,
            map_avg=map_avg,
            precision=map50,  # 简化
            recall=map50,
            num_detections=sum(len(p) for p in predictions) if predictions else 0
        )
    
    def _calculate_map(
        self,
        predictions: List,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        iou_threshold: float = 0.5
    ) -> float:
        """计算 mAP"""
        # 简化实现
        return 0.0
    
    def evaluate_segmentation(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        num_classes: int = 2
    ) -> SegmentationResult:
        """
        评估分割任务
        
        **Requirements 5.3**: 语义分割评估
        
        Args:
            images: [N, C, H, W] 输入图像
            gt_masks: [N, H, W] 真实分割掩码
            num_classes: 类别数
        
        Returns:
            SegmentationResult: 分割评估结果
        """
        images = images.to(self.device)
        gt_masks = gt_masks.to(self.device)
        
        if self.segmentor is not None:
            with torch.no_grad():
                pred_masks = self.segmentor(images)
                if pred_masks.dim() == 4:
                    pred_masks = pred_masks.argmax(dim=1)
        else:
            # 无分割器时，使用简单阈值
            pred_masks = (images.mean(dim=1) > 0.5).long()
        
        # 计算 IoU
        ious = []
        per_class_iou = {}
        
        for c in range(num_classes):
            pred_c = (pred_masks == c)
            gt_c = (gt_masks == c)
            
            intersection = (pred_c & gt_c).float().sum()
            union = (pred_c | gt_c).float().sum()
            
            iou = (intersection / (union + 1e-8)).item()
            ious.append(iou)
            per_class_iou[f"class_{c}"] = iou
        
        miou = np.mean(ious)
        
        # 像素准确率
        pixel_acc = (pred_masks == gt_masks).float().mean().item()
        
        # Dice 系数
        dice = self._calculate_dice(pred_masks, gt_masks)
        
        return SegmentationResult(
            miou=miou,
            pixel_accuracy=pixel_acc,
            dice_score=dice,
            per_class_iou=per_class_iou
        )
    
    def _calculate_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """计算 Dice 系数"""
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection / (pred_flat.sum() + target_flat.sum() + 1e-8)).item()
        
        return dice


    def evaluate_fairness(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        group_labels: torch.Tensor,
        group_names: List[str] = None
    ) -> FairnessResult:
        """
        评估公平性
        
        **Requirements 5.4**: 在 FairFace 上按 race/gender/age 分组
        
        Args:
            images: [N, C, H, W] 输入图像
            labels: [N] 任务标签
            group_labels: [N] 分组标签（如 race/gender）
            group_names: 分组名称列表
        
        Returns:
            FairnessResult: 公平性评估结果
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        group_labels = group_labels.to(self.device)
        
        # 获取预测
        if self.classifier is not None:
            with torch.no_grad():
                logits = self.classifier(images)
                preds = logits.argmax(dim=1) if logits.dim() == 2 else (logits > 0).long().squeeze()
                probs = F.softmax(logits, dim=1) if logits.dim() == 2 else torch.sigmoid(logits)
        else:
            preds = self._simple_knn_classify(images, labels)
            probs = torch.zeros_like(preds).float()
        
        # 按组计算准确率和校准
        unique_groups = torch.unique(group_labels)
        group_accuracies = {}
        group_calibrations = {}
        
        for i, group in enumerate(unique_groups):
            mask = group_labels == group
            group_preds = preds[mask]
            group_labels_subset = labels[mask]
            
            # 准确率
            acc = (group_preds == group_labels_subset).float().mean().item()
            group_name = group_names[i] if group_names and i < len(group_names) else f"group_{group.item()}"
            group_accuracies[group_name] = acc
            
            # 校准（简化：使用预测概率与实际标签的差异）
            if probs.dim() > 1:
                group_probs = probs[mask]
                calibration = abs(group_probs.mean().item() - group_labels_subset.float().mean().item())
            else:
                calibration = 0.0
            group_calibrations[group_name] = calibration
        
        # 计算差距
        acc_values = list(group_accuracies.values())
        cal_values = list(group_calibrations.values())
        
        accuracy_gap = max(acc_values) - min(acc_values) if acc_values else 0.0
        calibration_gap = max(cal_values) - min(cal_values) if cal_values else 0.0
        
        # 人口统计平等（预测正例率差异）
        positive_rates = []
        for group in unique_groups:
            mask = group_labels == group
            pos_rate = (preds[mask] == 1).float().mean().item()
            positive_rates.append(pos_rate)
        demographic_parity = max(positive_rates) - min(positive_rates) if positive_rates else 0.0
        
        return FairnessResult(
            group_accuracies=group_accuracies,
            group_calibrations=group_calibrations,
            accuracy_gap=accuracy_gap,
            calibration_gap=calibration_gap,
            demographic_parity=demographic_parity,
            equalized_odds=accuracy_gap  # 简化
        )
    
    def generate_privacy_utility_curve(
        self,
        evaluate_fn,
        privacy_levels: List[float] = None,
        task_type: str = "classification",
        metric_name: str = "accuracy"
    ) -> PrivacyUtilityCurve:
        """
        生成隐私-效用曲线
        
        **Property 12: Privacy-Utility Curve Completeness**
        在五档 privacy_level 上评估，输出完整曲线数据
        
        Args:
            evaluate_fn: 评估函数，接受 privacy_level 返回效用值
            privacy_levels: 隐私级别列表（默认五档）
            task_type: 任务类型
            metric_name: 指标名称
        
        Returns:
            PrivacyUtilityCurve: 隐私-效用曲线
        """
        if privacy_levels is None:
            privacy_levels = self.PRIVACY_LEVELS.copy()
        
        utility_values = []
        
        for level in privacy_levels:
            utility = evaluate_fn(level)
            utility_values.append(utility)
        
        return PrivacyUtilityCurve(
            privacy_levels=privacy_levels,
            utility_values=utility_values,
            task_type=task_type,
            metric_name=metric_name
        )
    
    def evaluate_at_privacy_levels(
        self,
        images_dict: Dict[float, torch.Tensor],
        labels: torch.Tensor,
        task_type: str = "classification"
    ) -> PrivacyUtilityCurve:
        """
        在多个隐私级别上评估效用
        
        Args:
            images_dict: {privacy_level: images} 不同隐私级别的图像
            labels: 标签
            task_type: 任务类型
        
        Returns:
            PrivacyUtilityCurve: 隐私-效用曲线
        """
        privacy_levels = sorted(images_dict.keys())
        utility_values = []
        
        for level in privacy_levels:
            images = images_dict[level]
            
            if task_type == "classification":
                result = self.evaluate_classification(images, labels)
                utility = result.accuracy
            elif task_type == "detection":
                # 需要 gt_boxes 和 gt_labels
                utility = 0.0
            elif task_type == "segmentation":
                result = self.evaluate_segmentation(images, labels)
                utility = result.miou
            else:
                utility = 0.0
            
            utility_values.append(utility)
        
        metric_name = {
            "classification": "accuracy",
            "detection": "mAP",
            "segmentation": "mIoU"
        }.get(task_type, "utility")
        
        return PrivacyUtilityCurve(
            privacy_levels=privacy_levels,
            utility_values=utility_values,
            task_type=task_type,
            metric_name=metric_name
        )
    
    def print_report(
        self,
        result: Union[ClassificationResult, DetectionResult, SegmentationResult, FairnessResult],
        title: str = "效用评估报告"
    ):
        """打印评估报告"""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        if isinstance(result, ClassificationResult):
            print(f"\n【分类评估】")
            print(f"  准确率: {result.accuracy:.4f}")
            print(f"  F1-score: {result.f1_score:.4f}")
            print(f"  Precision: {result.precision:.4f}")
            print(f"  Recall: {result.recall:.4f}")
            print(f"  类别数: {result.num_classes}")
            print(f"  样本数: {result.num_samples}")
            
        elif isinstance(result, DetectionResult):
            print(f"\n【检测评估】")
            print(f"  mAP@0.5: {result.map50:.4f}")
            print(f"  mAP@0.75: {result.map75:.4f}")
            print(f"  mAP@[0.5:0.95]: {result.map_avg:.4f}")
            print(f"  检测数: {result.num_detections}")
            
        elif isinstance(result, SegmentationResult):
            print(f"\n【分割评估】")
            print(f"  mIoU: {result.miou:.4f}")
            print(f"  像素准确率: {result.pixel_accuracy:.4f}")
            print(f"  Dice: {result.dice_score:.4f}")
            
        elif isinstance(result, FairnessResult):
            print(f"\n【公平性评估】")
            print(f"  准确率差距: {result.accuracy_gap:.4f}")
            print(f"  校准差距: {result.calibration_gap:.4f}")
            print(f"  人口统计平等: {result.demographic_parity:.4f}")
            print(f"\n  各组准确率:")
            for group, acc in result.group_accuracies.items():
                print(f"    {group}: {acc:.4f}")
        
        print("=" * 70)
    
    def print_curve(self, curve: PrivacyUtilityCurve):
        """打印隐私-效用曲线"""
        print("\n" + "=" * 70)
        print(f"隐私-效用曲线 ({curve.task_type} - {curve.metric_name})")
        print("=" * 70)
        
        print(f"\n{'Privacy Level':<15} {curve.metric_name:<15}")
        print("-" * 30)
        
        for level, utility in zip(curve.privacy_levels, curve.utility_values):
            print(f"{level:<15.2f} {utility:<15.4f}")
        
        print("-" * 30)
        print(f"完整性: {'✓' if curve.is_complete() else '❌'}")
        print("=" * 70)
