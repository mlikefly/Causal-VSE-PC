"""
Reconstruction Attack implementation.

attack_success = identity_similarity (per GC7)

**Validates: Requirements R2.AC5**
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
    """Reconstruction attack metrics."""
    identity_similarity: float  # attack_success
    psnr: float
    ssim: float
    mse: float
    sensitive_psnr: Optional[float] = None
    sensitive_ssim: Optional[float] = None


@AttackRegistry.register(AttackType.RECONSTRUCTION)
class ReconstructionAttack(AttackBase):
    """
    Reconstruction Attack.
    
    Evaluates whether an attacker can reconstruct original images
    from Z-view using decoder networks.
    
    attack_success = identity_similarity (lower is better for privacy)
    """
    
    attack_type = AttackType.RECONSTRUCTION
    
    def __init__(self, device: str = None, decoder=None, face_encoder=None):
        """
        Initialize reconstruction attack.
        
        Args:
            device: Compute device
            decoder: Reconstruction decoder (U-Net, VAE, etc.)
            face_encoder: Face encoder for identity similarity
        """
        super().__init__(device)
        self.decoder = decoder
        self.face_encoder = face_encoder
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train the reconstruction decoder.
        
        Args:
            ctx: Attack training context
            train_encrypted: Encrypted images (Z-view)
            train_original: Original images (for training)
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        # In practice, decoder would be trained here
        # For now, we assume pre-trained decoder
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        Evaluate reconstruction attack.
        
        Args:
            ctx: Evaluation context
            encrypted_images: Encrypted images (Z-view)
            original_images: Original images (ground truth)
            sensitive_mask: Optional mask for sensitive regions
            
        Returns:
            AttackResult with identity_similarity as attack_success
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
        """Reconstruct images from encrypted."""
        if self.decoder is not None:
            import torch
            with torch.no_grad():
                if hasattr(encrypted_images, 'to'):
                    encrypted_images = encrypted_images.to(self.device)
                reconstructed = self.decoder(encrypted_images)
                if hasattr(reconstructed, 'cpu'):
                    reconstructed = reconstructed.cpu().numpy()
            return reconstructed
        
        # Fallback: return encrypted as-is (worst case for attacker)
        if hasattr(encrypted_images, 'numpy'):
            return encrypted_images.numpy()
        return encrypted_images
    
    def _compute_metrics(
        self,
        encrypted_images,
        original_images,
        sensitive_mask=None
    ) -> ReconstructionMetrics:
        """Compute reconstruction metrics."""
        # Reconstruct
        reconstructed = self._reconstruct(encrypted_images)
        
        if hasattr(original_images, 'numpy'):
            original_images = original_images.numpy()
        
        # Compute identity similarity
        identity_sim = self._compute_identity_similarity(
            reconstructed, original_images
        )
        
        # Compute image quality metrics
        psnr = self._compute_psnr(reconstructed, original_images)
        ssim = self._compute_ssim(reconstructed, original_images)
        mse = np.mean((reconstructed - original_images) ** 2)
        
        # Sensitive region metrics
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
        """Compute identity similarity using face encoder."""
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
                
                # Cosine similarity
                recon_emb = recon_emb / (recon_emb.norm(dim=1, keepdim=True) + 1e-8)
                orig_emb = orig_emb / (orig_emb.norm(dim=1, keepdim=True) + 1e-8)
                
                similarity = (recon_emb * orig_emb).sum(dim=1).mean()
                return float(similarity.cpu().numpy())
        
        # Fallback: pixel-level cosine similarity
        recon_flat = reconstructed.reshape(len(reconstructed), -1)
        orig_flat = original.reshape(len(original), -1)
        
        similarities = []
        for r, o in zip(recon_flat, orig_flat):
            sim = np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-8)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _compute_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute PSNR."""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        max_val = max(pred.max(), target.max(), 1.0)
        return float(10 * np.log10(max_val ** 2 / mse))
    
    def _compute_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute simplified SSIM."""
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
