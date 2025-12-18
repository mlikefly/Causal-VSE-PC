"""
Face Verification Attack implementation.

attack_success = TAR@FAR=1e-3 (per GC7)

**Validates: Requirements R2.AC3**
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
    """Face verification attack metrics."""
    tar_at_far_1e3: float  # TAR@FAR=1e-3 (attack_success)
    auc: float
    eer: float  # Equal Error Rate
    num_pairs: int
    num_positive_pairs: int
    num_negative_pairs: int


@AttackRegistry.register(AttackType.FACE_VERIFICATION)
class FaceVerificationAttack(AttackBase):
    """
    Face Verification Attack.
    
    Evaluates whether an attacker can verify identity from Z-view.
    Uses embedding similarity to compute TAR@FAR.
    
    attack_success = TAR@FAR=1e-3 (lower is better for privacy)
    """
    
    attack_type = AttackType.FACE_VERIFICATION
    
    def __init__(self, device: str = None, encoder=None):
        """
        Initialize face verification attack.
        
        Args:
            device: Compute device
            encoder: Face encoder model (e.g., ArcFace, FaceNet)
        """
        super().__init__(device)
        self.encoder = encoder
        self.threshold: Optional[float] = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train/calibrate the attack.
        
        For face verification, this mainly involves:
        1. Computing embeddings for gallery images
        2. Finding optimal threshold (if needed)
        
        Args:
            ctx: Attack training context
            gallery_images: Gallery images for verification
            gallery_labels: Identity labels for gallery
        """
        self.validate_threat_level(ctx)
        self.fit_context = ctx
        
        gallery_images = kwargs.get('gallery_images')
        gallery_labels = kwargs.get('gallery_labels')
        
        if gallery_images is not None and self.encoder is not None:
            # Compute gallery embeddings
            self.gallery_embeddings = self._compute_embeddings(gallery_images)
            self.gallery_labels = gallery_labels
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        Evaluate face verification attack.
        
        Args:
            ctx: Evaluation context
            query_images: Query images (Z-view)
            query_labels: Identity labels for queries
            
        Returns:
            AttackResult with TAR@FAR=1e-3 as attack_success
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
        
        # Compute metrics
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
        """Compute face embeddings."""
        if self.encoder is None:
            # Fallback: use flattened pixels
            if hasattr(images, 'numpy'):
                images = images.numpy()
            return images.reshape(len(images), -1)
        
        # Use encoder
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
        """Compute face verification metrics."""
        # Compute query embeddings
        query_embeddings = self._compute_embeddings(query_images)
        
        # Get gallery (use query as gallery if not provided)
        gallery_embeddings = kwargs.get('gallery_embeddings', query_embeddings)
        gallery_labels = kwargs.get('gallery_labels', query_labels)
        
        if hasattr(gallery_labels, 'numpy'):
            gallery_labels = gallery_labels.numpy()
        if hasattr(query_labels, 'numpy'):
            query_labels = query_labels.numpy()
        
        # Compute similarity scores and labels
        scores = []
        labels = []
        
        for i in range(len(query_embeddings)):
            for j in range(len(gallery_embeddings)):
                if i == j and gallery_embeddings is query_embeddings:
                    continue  # Skip self-comparison
                
                # Cosine similarity
                sim = np.dot(query_embeddings[i], gallery_embeddings[j])
                sim /= (np.linalg.norm(query_embeddings[i]) * 
                       np.linalg.norm(gallery_embeddings[j]) + 1e-8)
                
                scores.append(sim)
                labels.append(1 if query_labels[i] == gallery_labels[j] else 0)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Compute TAR@FAR=1e-3
        tar_at_far = self._compute_tar_at_far(scores, labels, far_target=1e-3)
        
        # Compute AUC
        auc = self._compute_auc(scores, labels)
        
        # Compute EER
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
        """Compute TAR at specified FAR."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
        
        # Find threshold for target FAR
        thresholds = np.sort(neg_scores)[::-1]
        
        for thresh in thresholds:
            far = np.mean(neg_scores >= thresh)
            if far <= far_target:
                tar = np.mean(pos_scores >= thresh)
                return float(tar)
        
        return 0.0
    
    def _compute_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC using trapezoidal rule."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        # Simple AUC computation
        correct = 0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    correct += 1
                elif ps == ns:
                    correct += 0.5
        
        return correct / (len(pos_scores) * len(neg_scores))
    
    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Equal Error Rate."""
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
