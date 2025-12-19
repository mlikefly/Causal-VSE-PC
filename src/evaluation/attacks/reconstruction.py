"""
重建攻击实现。

attack_success = identity_similarity（按 GC7）

**验证: 需求 R2.AC5**
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
class ReconstructionMetrics:
    """重建攻击指标。"""
    identity_similarity: float  # attack_success
    psnr: float
    ssim: float
    mse: float
    sensitive_psnr: Optional[float] = None
    sensitive_ssim: Optional[float] = None


@AttackRegistry.register(AttackType.RECONSTRUCTION)
class ReconstructionAttack(AttackBase):
    """
    重建攻击。
    
    评估攻击者是否能从 Z-view 使用解码器网络重建原始图像。
    
    attack_success = identity_similarity（越低隐私保护越好）
    """
    
    attack_type = AttackType.RECONSTRUCTION
    
    def __init__(self, device: str = None, decoder=None, face_encoder=None):
        """
        初始化重建攻击。
        
        参数:
            device: 计算设备
            decoder: 重建解码器（U-Net、VAE 等）
            face_encoder: 用于身份相似度的人脸编码器
        """
        super().__init__(device)
        self.decoder = decoder
        self.face_encoder = face_encoder
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练重建解码器。
        
        参数:
            ctx: 攻击训练上下文
            train_encrypted: 加密图像（Z-view）
            train_original: 原始图像（用于训练）
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        # 实际中解码器会在这里训练
        # 目前假设使用预训练的解码器
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估重建攻击。
        
        参数:
            ctx: 评估上下文
            encrypted_images: 加密图像（Z-view）
            original_images: 原始图像（真值）
            sensitive_mask: 可选的敏感区域掩码
            
        返回:
            以 identity_similarity 作为 attack_success 的 AttackResult
        """
        encrypted_images = kwargs.get('encrypted_images')
        original_images = kwargs.get('original_images')
        sensitive_mask = kwargs.get('sensitive_mask')
        
        if encrypted_images is None or original_images is None:
            return AttackResult(
                attack_type=self.attack_type,
                threat_level=ctx.threat_level,
                attack_success=0.0,
                metric_name="identity_similarity",
                metric_value=0.0,
                status="failed",
            )
        
        metrics = self._compute_metrics(
            encrypted_images, original_images, sensitive_mask
        )
        
        return AttackResult(
            attack_type=self.attack_type,
            threat_level=ctx.threat_level,
            attack_success=metrics.identity_similarity,
            metric_name="identity_similarity",
            metric_value=metrics.identity_similarity,
            status="success",
            additional_metrics={
                'psnr': metrics.psnr,
                'ssim': metrics.ssim,
                'mse': metrics.mse,
            }
        )
    
    def _reconstruct(self, encrypted_images) -> np.ndarray:
        """从加密图像重建。"""
        if self.decoder is not None:
            import torch
            with torch.no_grad():
                if hasattr(encrypted_images, 'to'):
                    encrypted_images = encrypted_images.to(self.device)
                reconstructed = self.decoder(encrypted_images)
                if hasattr(reconstructed, 'cpu'):
                    reconstructed = reconstructed.cpu().numpy()
            return reconstructed
        
        # 回退：直接返回加密图像（攻击者最坏情况）
        if hasattr(encrypted_images, 'numpy'):
            return encrypted_images.numpy()
        return encrypted_images
    
    def _compute_metrics(
        self,
        encrypted_images,
        original_images,
        sensitive_mask=None
    ) -> ReconstructionMetrics:
        """计算重建指标。"""
        # 重建
        reconstructed = self._reconstruct(encrypted_images)
        
        if hasattr(original_images, 'numpy'):
            original_images = original_images.numpy()
        
        # 计算身份相似度
        identity_sim = self._compute_identity_similarity(
            reconstructed, original_images
        )
        
        # 计算图像质量指标
        psnr = self._compute_psnr(reconstructed, original_images)
        ssim = self._compute_ssim(reconstructed, original_images)
        mse = np.mean((reconstructed - original_images) ** 2)
        
        # 敏感区域指标
        sensitive_psnr = None
        sensitive_ssim = None
        if sensitive_mask is not None:
            if hasattr(sensitive_mask, 'numpy'):
                sensitive_mask = sensitive_mask.numpy()
            sensitive_psnr = self._compute_psnr(
                reconstructed * sensitive_mask,
                original_images * sensitive_mask
            )
            sensitive_ssim = self._compute_ssim(
                reconstructed * sensitive_mask,
                original_images * sensitive_mask
            )
        
        return ReconstructionMetrics(
            identity_similarity=identity_sim,
            psnr=psnr,
            ssim=ssim,
            mse=mse,
            sensitive_psnr=sensitive_psnr,
            sensitive_ssim=sensitive_ssim,
        )
    
    def _compute_identity_similarity(
        self,
        reconstructed: np.ndarray,
        original: np.ndarray
    ) -> float:
        """使用人脸编码器计算身份相似度。"""
        if self.face_encoder is not None:
            import torch
            with torch.no_grad():
                recon_tensor = torch.from_numpy(reconstructed).float()
                orig_tensor = torch.from_numpy(original).float()
                
                if hasattr(recon_tensor, 'to'):
                    recon_tensor = recon_tensor.to(self.device)
                    orig_tensor = orig_tensor.to(self.device)
                
                recon_emb = self.face_encoder(recon_tensor)
                orig_emb = self.face_encoder(orig_tensor)
                
                # 余弦相似度
                recon_emb = recon_emb / (recon_emb.norm(dim=1, keepdim=True) + 1e-8)
                orig_emb = orig_emb / (orig_emb.norm(dim=1, keepdim=True) + 1e-8)
                
                similarity = (recon_emb * orig_emb).sum(dim=1).mean()
                return float(similarity.cpu().numpy())
        
        # 回退：像素级余弦相似度
        recon_flat = reconstructed.reshape(len(reconstructed), -1)
        orig_flat = original.reshape(len(original), -1)
        
        similarities = []
        for r, o in zip(recon_flat, orig_flat):
            sim = np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-8)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _compute_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算 PSNR。"""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        max_val = max(pred.max(), target.max(), 1.0)
        return float(10 * np.log10(max_val ** 2 / mse))
    
    def _compute_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算简化的 SSIM。"""
        pred = pred.astype(np.float64)
        target = target.astype(np.float64)
        
        mu_pred = pred.mean()
        mu_target = target.mean()
        
        sigma_pred = pred.var()
        sigma_target = target.var()
        sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        
        return float(ssim)
