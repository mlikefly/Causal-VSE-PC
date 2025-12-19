"""
人脸验证攻击实现。

attack_success = TAR@FAR=1e-3 (按 GC7)

**验证: 需求 R2.AC3**
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
class FaceVerificationMetrics:
    """人脸验证攻击指标。"""
    tar_at_far_1e3: float  # TAR@FAR=1e-3 (attack_success)
    auc: float
    eer: float  # 等错误率
    num_pairs: int
    num_positive_pairs: int
    num_negative_pairs: int


@AttackRegistry.register(AttackType.FACE_VERIFICATION)
class FaceVerificationAttack(AttackBase):
    """
    人脸验证攻击。
    
    评估攻击者是否能从 Z-view 验证身份。
    使用嵌入相似度计算 TAR@FAR。
    
    attack_success = TAR@FAR=1e-3（越低隐私保护越好）
    """
    
    attack_type = AttackType.FACE_VERIFICATION
    
    def __init__(self, device: str = None, encoder=None):
        """
        初始化人脸验证攻击。
        
        参数:
            device: 计算设备
            encoder: 人脸编码器模型（如 ArcFace、FaceNet）
        """
        super().__init__(device)
        self.encoder = encoder
        self.threshold: Optional[float] = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练/校准攻击。
        
        对于人脸验证，主要包括:
        1. 计算图库图像的嵌入
        2. 找到最优阈值（如需要）
        
        参数:
            ctx: 攻击训练上下文
            gallery_images: 用于验证的图库图像
            gallery_labels: 图库的身份标签
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        gallery_images = kwargs.get('gallery_images')
        gallery_labels = kwargs.get('gallery_labels')
        
        if gallery_images is not None and self.encoder is not None:
            # 计算图库嵌入
            self.gallery_embeddings = self._compute_embeddings(gallery_images)
            self.gallery_labels = gallery_labels
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估人脸验证攻击。
        
        参数:
            ctx: 评估上下文
            query_images: 查询图像（Z-view）
            query_labels: 查询的身份标签
            
        返回:
            以 TAR@FAR=1e-3 作为 attack_success 的 AttackResult
        """
        query_images = kwargs.get('query_images')
        query_labels = kwargs.get('query_labels')
        
        if query_images is None or query_labels is None:
            return AttackResult(
                attack_type=self.attack_type,
                threat_level=ctx.threat_level,
                attack_success=0.0,
                metric_name="TAR@FAR=1e-3",
                metric_value=0.0,
                status="failed",
            )
        
        # 计算指标
        metrics = self._compute_verification_metrics(
            query_images, query_labels, **kwargs
        )
        
        return AttackResult(
            attack_type=self.attack_type,
            threat_level=ctx.threat_level,
            attack_success=metrics.tar_at_far_1e3,
            metric_name="TAR@FAR=1e-3",
            metric_value=metrics.tar_at_far_1e3,
            status="success",
            additional_metrics={
                'auc': metrics.auc,
                'eer': metrics.eer,
                'num_pairs': metrics.num_pairs,
            }
        )
    
    def _compute_embeddings(self, images) -> np.ndarray:
        """计算人脸嵌入。"""
        if self.encoder is None:
            # 回退：使用展平的像素
            if hasattr(images, 'numpy'):
                images = images.numpy()
            return images.reshape(len(images), -1)
        
        # 使用编码器
        import torch
        with torch.no_grad():
            if hasattr(images, 'to'):
                images = images.to(self.device)
            embeddings = self.encoder(images)
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
        return embeddings
    
    def _compute_verification_metrics(
        self,
        query_images,
        query_labels,
        **kwargs
    ) -> FaceVerificationMetrics:
        """计算人脸验证指标。"""
        # 计算查询嵌入
        query_embeddings = self._compute_embeddings(query_images)
        
        # 获取图库（如果未提供则使用查询作为图库）
        gallery_embeddings = kwargs.get('gallery_embeddings', query_embeddings)
        gallery_labels = kwargs.get('gallery_labels', query_labels)
        
        if hasattr(gallery_labels, 'numpy'):
            gallery_labels = gallery_labels.numpy()
        if hasattr(query_labels, 'numpy'):
            query_labels = query_labels.numpy()
        
        # 计算相似度分数和标签
        scores = []
        labels = []
        
        for i in range(len(query_embeddings)):
            for j in range(len(gallery_embeddings)):
                if i == j and gallery_embeddings is query_embeddings:
                    continue  # 跳过自比较
                
                # 余弦相似度
                sim = np.dot(query_embeddings[i], gallery_embeddings[j])
                sim /= (np.linalg.norm(query_embeddings[i]) * 
                       np.linalg.norm(gallery_embeddings[j]) + 1e-8)
                
                scores.append(sim)
                labels.append(1 if query_labels[i] == gallery_labels[j] else 0)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # 计算 TAR@FAR=1e-3
        tar_at_far = self._compute_tar_at_far(scores, labels, far_target=1e-3)
        
        # 计算 AUC
        auc = self._compute_auc(scores, labels)
        
        # 计算 EER
        eer = self._compute_eer(scores, labels)
        
        return FaceVerificationMetrics(
            tar_at_far_1e3=tar_at_far,
            auc=auc,
            eer=eer,
            num_pairs=len(scores),
            num_positive_pairs=int(labels.sum()),
            num_negative_pairs=int((1 - labels).sum()),
        )
    
    def _compute_tar_at_far(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        far_target: float = 1e-3
    ) -> float:
        """计算指定 FAR 下的 TAR。"""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
        
        # 找到目标 FAR 的阈值
        thresholds = np.sort(neg_scores)[::-1]
        
        for thresh in thresholds:
            far = np.mean(neg_scores >= thresh)
            if far <= far_target:
                tar = np.mean(pos_scores >= thresh)
                return float(tar)
        
        return 0.0
    
    def _compute_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """使用梯形法则计算 AUC。"""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        # 简单 AUC 计算
        correct = 0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    correct += 1
                elif ps == ns:
                    correct += 0.5
        
        return correct / (len(pos_scores) * len(neg_scores))
    
    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """计算等错误率。"""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        
        min_diff = float('inf')
        eer = 0.5
        
        for thresh in thresholds:
            far = np.mean(neg_scores >= thresh)
            frr = np.mean(pos_scores < thresh)
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
        
        return float(eer)
