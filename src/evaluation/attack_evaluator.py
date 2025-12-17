# -*- coding: utf-8 -*-
"""
攻击评估器 (Attack Evaluator)

实现四类攻击评估：
1. 身份识别攻击 (Identity Recognition Attack)
2. 重建攻击 (Reconstruction Attack)
3. 属性推断攻击 (Attribute Inference Attack)
4. 可链接性攻击 (Linkability Attack)

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IdentityAttackResult:
    """身份识别攻击结果"""
    top1_accuracy: float = 0.0  # Top-1 识别率
    top5_accuracy: float = 0.0  # Top-5 识别率
    num_identities: int = 0     # 身份数量
    num_samples: int = 0        # 样本数量
    embedding_dim: int = 0      # 嵌入维度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'top1_accuracy': self.top1_accuracy,
            'top5_accuracy': self.top5_accuracy,
            'num_identities': self.num_identities,
            'num_samples': self.num_samples,
            'embedding_dim': self.embedding_dim
        }


@dataclass
class ReconstructionAttackResult:
    """重建攻击结果"""
    psnr: float = 0.0           # 峰值信噪比 (dB)
    ssim: float = 0.0           # 结构相似性
    lpips: float = 0.0          # 感知损失
    mse: float = 0.0            # 均方误差
    sensitive_psnr: float = 0.0 # 敏感区域 PSNR
    sensitive_ssim: float = 0.0 # 敏感区域 SSIM
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'psnr': self.psnr,
            'ssim': self.ssim,
            'lpips': self.lpips,
            'mse': self.mse,
            'sensitive_psnr': self.sensitive_psnr,
            'sensitive_ssim': self.sensitive_ssim
        }


@dataclass
class AttributeInferenceResult:
    """属性推断攻击结果"""
    accuracy_original: float = 0.0      # 原始图像准确率
    accuracy_encrypted: float = 0.0     # 加密图像准确率
    accuracy_drop: float = 0.0          # 准确率下降
    attribute_name: str = ""            # 属性名称
    num_classes: int = 0                # 类别数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy_original': self.accuracy_original,
            'accuracy_encrypted': self.accuracy_encrypted,
            'accuracy_drop': self.accuracy_drop,
            'attribute_name': self.attribute_name,
            'num_classes': self.num_classes
        }


@dataclass
class LinkabilityAttackResult:
    """可链接性攻击结果"""
    linkage_auc: float = 0.0            # 链接 AUC
    intra_class_distance: float = 0.0   # 类内距离
    inter_class_distance: float = 0.0   # 类间距离
    distance_ratio: float = 0.0         # 距离比
    num_identities: int = 0             # 身份数量
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'linkage_auc': self.linkage_auc,
            'intra_class_distance': self.intra_class_distance,
            'inter_class_distance': self.inter_class_distance,
            'distance_ratio': self.distance_ratio,
            'num_identities': self.num_identities
        }


@dataclass
class AttackEvaluationMatrix:
    """
    攻击评估矩阵
    
    **Property 11: Attack Evaluation Matrix Completeness**
    包含四类攻击的完整评估结果
    """
    identity: Optional[IdentityAttackResult] = None
    reconstruction: Optional[ReconstructionAttackResult] = None
    attribute_inference: Dict[str, AttributeInferenceResult] = field(default_factory=dict)
    linkability: Optional[LinkabilityAttackResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.identity:
            result['identity'] = self.identity.to_dict()
        if self.reconstruction:
            result['reconstruction'] = self.reconstruction.to_dict()
        if self.attribute_inference:
            result['attribute_inference'] = {
                k: v.to_dict() for k, v in self.attribute_inference.items()
            }
        if self.linkability:
            result['linkability'] = self.linkability.to_dict()
        return result
    
    def is_complete(self) -> bool:
        """检查评估矩阵是否完整"""
        return all([
            self.identity is not None,
            self.reconstruction is not None,
            len(self.attribute_inference) > 0,
            self.linkability is not None
        ])


class AttackEvaluator:
    """
    攻击评估器
    
    支持四类攻击评估：
    1. 身份识别攻击 - 使用 ArcFace/FaceNet 编码器
    2. 重建攻击 - 使用 U-Net/VAE 重建器
    3. 属性推断攻击 - 预测敏感属性
    4. 可链接性攻击 - 计算 embedding 聚类可分性
    """
    
    def __init__(
        self,
        device: str = None,
        face_encoder: nn.Module = None,
        reconstructor: nn.Module = None,
        attribute_classifier: nn.Module = None
    ):
        """
        初始化攻击评估器
        
        Args:
            device: 计算设备
            face_encoder: 人脸编码器（用于身份识别和可链接性攻击）
            reconstructor: 重建器（用于重建攻击）
            attribute_classifier: 属性分类器（用于属性推断攻击）
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_encoder = face_encoder
        self.reconstructor = reconstructor
        self.attribute_classifier = attribute_classifier
        
        # 移动模型到设备
        if self.face_encoder:
            self.face_encoder = self.face_encoder.to(self.device)
            self.face_encoder.eval()
        if self.reconstructor:
            self.reconstructor = self.reconstructor.to(self.device)
            self.reconstructor.eval()
        if self.attribute_classifier:
            self.attribute_classifier = self.attribute_classifier.to(self.device)
            self.attribute_classifier.eval()
    
    def evaluate_identity_attack(
        self,
        encrypted_images: torch.Tensor,
        identity_labels: torch.Tensor,
        gallery_embeddings: torch.Tensor = None,
        gallery_labels: torch.Tensor = None
    ) -> IdentityAttackResult:
        """
        评估身份识别攻击
        
        **Requirements 6.1**: 使用 ArcFace/FaceNet 编码器，输出 Top-1/Top-5 识别率
        
        Args:
            encrypted_images: [N, C, H, W] 加密图像
            identity_labels: [N] 身份标签
            gallery_embeddings: [M, D] 画廊嵌入（可选）
            gallery_labels: [M] 画廊标签（可选）
        
        Returns:
            IdentityAttackResult: 身份识别攻击结果
        """
        if self.face_encoder is None:
            # 使用简单的 CNN 作为默认编码器
            return self._evaluate_identity_simple(encrypted_images, identity_labels)
        
        encrypted_images = encrypted_images.to(self.device)
        
        with torch.no_grad():
            # 提取查询嵌入
            query_embeddings = self.face_encoder(encrypted_images)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
            # 如果没有提供画廊，使用查询集自身
            if gallery_embeddings is None:
                gallery_embeddings = query_embeddings
                gallery_labels = identity_labels
            
            gallery_embeddings = gallery_embeddings.to(self.device)
            gallery_labels = gallery_labels.to(self.device)
            
            # 计算相似度矩阵
            similarity = torch.mm(query_embeddings, gallery_embeddings.t())
            
            # Top-1 和 Top-5 准确率
            top1_correct = 0
            top5_correct = 0
            
            for i in range(len(encrypted_images)):
                # 排除自身
                sim = similarity[i].clone()
                if gallery_embeddings is query_embeddings:
                    sim[i] = -float('inf')
                
                # Top-5 索引
                _, top5_idx = sim.topk(5)
                top5_labels = gallery_labels[top5_idx]
                
                if identity_labels[i] == top5_labels[0]:
                    top1_correct += 1
                if identity_labels[i] in top5_labels:
                    top5_correct += 1
        
        n_samples = len(encrypted_images)
        n_identities = len(torch.unique(identity_labels))
        
        return IdentityAttackResult(
            top1_accuracy=top1_correct / n_samples,
            top5_accuracy=top5_correct / n_samples,
            num_identities=n_identities,
            num_samples=n_samples,
            embedding_dim=query_embeddings.shape[1]
        )
    
    def _evaluate_identity_simple(
        self,
        encrypted_images: torch.Tensor,
        identity_labels: torch.Tensor
    ) -> IdentityAttackResult:
        """简单的身份识别评估（无预训练编码器）"""
        # 使用图像像素作为特征
        features = encrypted_images.view(encrypted_images.shape[0], -1)
        features = F.normalize(features.float(), p=2, dim=1)
        
        # 计算相似度
        similarity = torch.mm(features, features.t())
        
        top1_correct = 0
        top5_correct = 0
        n_samples = len(encrypted_images)
        
        for i in range(n_samples):
            sim = similarity[i].clone()
            sim[i] = -float('inf')  # 排除自身
            
            _, top5_idx = sim.topk(min(5, n_samples - 1))
            top5_labels = identity_labels[top5_idx]
            
            if len(top5_labels) > 0 and identity_labels[i] == top5_labels[0]:
                top1_correct += 1
            if identity_labels[i] in top5_labels:
                top5_correct += 1
        
        return IdentityAttackResult(
            top1_accuracy=top1_correct / n_samples if n_samples > 0 else 0,
            top5_accuracy=top5_correct / n_samples if n_samples > 0 else 0,
            num_identities=len(torch.unique(identity_labels)),
            num_samples=n_samples,
            embedding_dim=features.shape[1]
        )
    
    def evaluate_reconstruction_attack(
        self,
        encrypted_images: torch.Tensor,
        original_images: torch.Tensor,
        sensitive_mask: torch.Tensor = None
    ) -> ReconstructionAttackResult:
        """
        评估重建攻击
        
        **Requirements 6.2**: 支持 U-Net/VAE 重建器，输出敏感区域 PSNR/SSIM/LPIPS
        
        Args:
            encrypted_images: [N, C, H, W] 加密图像
            original_images: [N, C, H, W] 原始图像
            sensitive_mask: [N, 1, H, W] 敏感区域掩码（可选）
        
        Returns:
            ReconstructionAttackResult: 重建攻击结果
        """
        encrypted_images = encrypted_images.to(self.device)
        original_images = original_images.to(self.device)
        
        # 尝试重建
        if self.reconstructor is not None:
            with torch.no_grad():
                reconstructed = self.reconstructor(encrypted_images)
        else:
            # 无重建器时，直接比较加密图像与原始图像
            reconstructed = encrypted_images
        
        # 计算全局指标
        mse = F.mse_loss(reconstructed, original_images).item()
        psnr = self._calculate_psnr(reconstructed, original_images)
        ssim = self._calculate_ssim(reconstructed, original_images)
        
        # 计算敏感区域指标
        if sensitive_mask is not None:
            sensitive_mask = sensitive_mask.to(self.device)
            sensitive_psnr = self._calculate_psnr(
                reconstructed * sensitive_mask,
                original_images * sensitive_mask
            )
            sensitive_ssim = self._calculate_ssim(
                reconstructed * sensitive_mask,
                original_images * sensitive_mask
            )
        else:
            sensitive_psnr = psnr
            sensitive_ssim = ssim
        
        # LPIPS（如果可用）
        lpips = self._calculate_lpips(reconstructed, original_images)
        
        return ReconstructionAttackResult(
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
            mse=mse,
            sensitive_psnr=sensitive_psnr,
            sensitive_ssim=sensitive_ssim
        )
    
    def _calculate_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """计算 PSNR"""
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return float('inf')
        return 10 * np.log10(max_val ** 2 / mse)
    
    def _calculate_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """计算简化版 SSIM"""
        # 简化实现：使用均值和方差
        pred = pred.float()
        target = target.float()
        
        mu_pred = pred.mean()
        mu_target = target.mean()
        
        sigma_pred = pred.var()
        sigma_target = target.var()
        sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        
        return ssim.item()
    
    def _calculate_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """计算 LPIPS（简化版，使用 L2 距离）"""
        # 简化实现：使用归一化 L2 距离
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        lpips = F.mse_loss(pred_flat, target_flat).item()
        return lpips


    def evaluate_attribute_inference(
        self,
        original_images: torch.Tensor,
        encrypted_images: torch.Tensor,
        attribute_labels: torch.Tensor,
        attribute_name: str = "unknown"
    ) -> AttributeInferenceResult:
        """
        评估属性推断攻击
        
        **Requirements 6.3**: 预测敏感属性（race/gender），输出准确率下降
        
        Args:
            original_images: [N, C, H, W] 原始图像
            encrypted_images: [N, C, H, W] 加密图像
            attribute_labels: [N] 属性标签
            attribute_name: 属性名称
        
        Returns:
            AttributeInferenceResult: 属性推断攻击结果
        """
        original_images = original_images.to(self.device)
        encrypted_images = encrypted_images.to(self.device)
        attribute_labels = attribute_labels.to(self.device)
        
        if self.attribute_classifier is not None:
            with torch.no_grad():
                # 原始图像预测
                orig_logits = self.attribute_classifier(original_images)
                orig_preds = orig_logits.argmax(dim=1)
                orig_acc = (orig_preds == attribute_labels).float().mean().item()
                
                # 加密图像预测
                enc_logits = self.attribute_classifier(encrypted_images)
                enc_preds = enc_logits.argmax(dim=1)
                enc_acc = (enc_preds == attribute_labels).float().mean().item()
        else:
            # 无分类器时，使用简单的特征匹配
            orig_acc, enc_acc = self._simple_attribute_inference(
                original_images, encrypted_images, attribute_labels
            )
        
        return AttributeInferenceResult(
            accuracy_original=orig_acc,
            accuracy_encrypted=enc_acc,
            accuracy_drop=orig_acc - enc_acc,
            attribute_name=attribute_name,
            num_classes=len(torch.unique(attribute_labels))
        )
    
    def _simple_attribute_inference(
        self,
        original_images: torch.Tensor,
        encrypted_images: torch.Tensor,
        attribute_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """简单的属性推断（无预训练分类器）"""
        # 使用 k-NN 分类
        k = 3
        
        # 原始图像特征
        orig_features = original_images.view(original_images.shape[0], -1).float()
        orig_features = F.normalize(orig_features, p=2, dim=1)
        
        # 加密图像特征
        enc_features = encrypted_images.view(encrypted_images.shape[0], -1).float()
        enc_features = F.normalize(enc_features, p=2, dim=1)
        
        # 留一法交叉验证
        orig_correct = 0
        enc_correct = 0
        n_samples = len(attribute_labels)
        
        for i in range(n_samples):
            # 原始图像 k-NN
            orig_sim = torch.mm(orig_features[i:i+1], orig_features.t())[0]
            orig_sim[i] = -float('inf')
            _, orig_top_k = orig_sim.topk(k)
            orig_pred = attribute_labels[orig_top_k].mode().values
            if orig_pred == attribute_labels[i]:
                orig_correct += 1
            
            # 加密图像 k-NN
            enc_sim = torch.mm(enc_features[i:i+1], enc_features.t())[0]
            enc_sim[i] = -float('inf')
            _, enc_top_k = enc_sim.topk(k)
            enc_pred = attribute_labels[enc_top_k].mode().values
            if enc_pred == attribute_labels[i]:
                enc_correct += 1
        
        return orig_correct / n_samples, enc_correct / n_samples
    
    def evaluate_linkability_attack(
        self,
        encrypted_images: torch.Tensor,
        identity_labels: torch.Tensor
    ) -> LinkabilityAttackResult:
        """
        评估可链接性攻击
        
        **Requirements 6.4**: 计算同身份跨样本的 embedding 聚类可分性，
        输出 linkage AUC 和类内/类间距离
        
        Args:
            encrypted_images: [N, C, H, W] 加密图像
            identity_labels: [N] 身份标签
        
        Returns:
            LinkabilityAttackResult: 可链接性攻击结果
        """
        encrypted_images = encrypted_images.to(self.device)
        
        # 提取嵌入
        if self.face_encoder is not None:
            with torch.no_grad():
                embeddings = self.face_encoder(encrypted_images)
                embeddings = F.normalize(embeddings, p=2, dim=1)
        else:
            # 使用像素特征
            embeddings = encrypted_images.view(encrypted_images.shape[0], -1).float()
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 计算类内和类间距离
        intra_distances = []
        inter_distances = []
        
        unique_ids = torch.unique(identity_labels)
        
        for i in range(len(encrypted_images)):
            for j in range(i + 1, len(encrypted_images)):
                dist = dist_matrix[i, j].item()
                if identity_labels[i] == identity_labels[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        intra_mean = np.mean(intra_distances) if intra_distances else 0
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        
        # 计算距离比
        distance_ratio = intra_mean / (inter_mean + 1e-8)
        
        # 计算 linkage AUC
        linkage_auc = self._calculate_linkage_auc(
            intra_distances, inter_distances
        )
        
        return LinkabilityAttackResult(
            linkage_auc=linkage_auc,
            intra_class_distance=intra_mean,
            inter_class_distance=inter_mean,
            distance_ratio=distance_ratio,
            num_identities=len(unique_ids)
        )
    
    def _calculate_linkage_auc(
        self,
        intra_distances: List[float],
        inter_distances: List[float]
    ) -> float:
        """计算 linkage AUC"""
        if not intra_distances or not inter_distances:
            return 0.5
        
        # 简化的 AUC 计算
        # 正样本：类内距离（应该小）
        # 负样本：类间距离（应该大）
        
        # 将距离转换为相似度分数（距离越小，分数越高）
        all_distances = intra_distances + inter_distances
        max_dist = max(all_distances) + 1e-8
        
        intra_scores = [1 - d / max_dist for d in intra_distances]
        inter_scores = [1 - d / max_dist for d in inter_distances]
        
        # 计算 AUC
        n_pos = len(intra_scores)
        n_neg = len(inter_scores)
        
        correct = 0
        for pos_score in intra_scores:
            for neg_score in inter_scores:
                if pos_score > neg_score:
                    correct += 1
                elif pos_score == neg_score:
                    correct += 0.5
        
        auc = correct / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5
        return auc
    
    def evaluate_all(
        self,
        original_images: torch.Tensor,
        encrypted_images: torch.Tensor,
        identity_labels: torch.Tensor = None,
        attribute_labels: Dict[str, torch.Tensor] = None,
        sensitive_mask: torch.Tensor = None
    ) -> AttackEvaluationMatrix:
        """
        执行完整的攻击评估
        
        **Property 11: Attack Evaluation Matrix Completeness**
        
        Args:
            original_images: [N, C, H, W] 原始图像
            encrypted_images: [N, C, H, W] 加密图像
            identity_labels: [N] 身份标签（可选）
            attribute_labels: {attr_name: [N]} 属性标签字典（可选）
            sensitive_mask: [N, 1, H, W] 敏感区域掩码（可选）
        
        Returns:
            AttackEvaluationMatrix: 完整的攻击评估矩阵
        """
        matrix = AttackEvaluationMatrix()
        
        # 1. 身份识别攻击
        if identity_labels is not None:
            matrix.identity = self.evaluate_identity_attack(
                encrypted_images, identity_labels
            )
        
        # 2. 重建攻击
        matrix.reconstruction = self.evaluate_reconstruction_attack(
            encrypted_images, original_images, sensitive_mask
        )
        
        # 3. 属性推断攻击
        if attribute_labels is not None:
            for attr_name, labels in attribute_labels.items():
                matrix.attribute_inference[attr_name] = self.evaluate_attribute_inference(
                    original_images, encrypted_images, labels, attr_name
                )
        
        # 4. 可链接性攻击
        if identity_labels is not None:
            matrix.linkability = self.evaluate_linkability_attack(
                encrypted_images, identity_labels
            )
        
        return matrix
    
    def print_report(self, matrix: AttackEvaluationMatrix):
        """打印攻击评估报告"""
        print("\n" + "=" * 70)
        print("攻击评估报告")
        print("=" * 70)
        
        # 身份识别攻击
        if matrix.identity:
            print("\n【身份识别攻击】")
            print(f"  Top-1 识别率: {matrix.identity.top1_accuracy:.4f}")
            print(f"  Top-5 识别率: {matrix.identity.top5_accuracy:.4f}")
            print(f"  身份数量: {matrix.identity.num_identities}")
            print(f"  样本数量: {matrix.identity.num_samples}")
        
        # 重建攻击
        if matrix.reconstruction:
            print("\n【重建攻击】")
            print(f"  PSNR: {matrix.reconstruction.psnr:.2f} dB")
            print(f"  SSIM: {matrix.reconstruction.ssim:.4f}")
            print(f"  LPIPS: {matrix.reconstruction.lpips:.4f}")
            print(f"  敏感区域 PSNR: {matrix.reconstruction.sensitive_psnr:.2f} dB")
            print(f"  敏感区域 SSIM: {matrix.reconstruction.sensitive_ssim:.4f}")
        
        # 属性推断攻击
        if matrix.attribute_inference:
            print("\n【属性推断攻击】")
            for attr_name, result in matrix.attribute_inference.items():
                print(f"  {attr_name}:")
                print(f"    原始准确率: {result.accuracy_original:.4f}")
                print(f"    加密准确率: {result.accuracy_encrypted:.4f}")
                print(f"    准确率下降: {result.accuracy_drop:.4f}")
        
        # 可链接性攻击
        if matrix.linkability:
            print("\n【可链接性攻击】")
            print(f"  Linkage AUC: {matrix.linkability.linkage_auc:.4f}")
            print(f"  类内距离: {matrix.linkability.intra_class_distance:.4f}")
            print(f"  类间距离: {matrix.linkability.inter_class_distance:.4f}")
            print(f"  距离比: {matrix.linkability.distance_ratio:.4f}")
        
        # 完整性检查
        print("\n" + "=" * 70)
        print(f"评估矩阵完整性: {'✓ 完整' if matrix.is_complete() else '⚠️ 不完整'}")
        print("=" * 70)
