"""
Attribute Inference Attack implementation.

attack_success = AUC (per GC7)

**Validates: Requirements R2.AC4**
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
    """Attribute inference attack metrics."""
    auc: float  # attack_success
    accuracy: float
    attribute_name: str
    num_classes: int
    num_samples: int


@AttackRegistry.register(AttackType.ATTRIBUTE_INFERENCE)
class AttributeInferenceAttack(AttackBase):
    """
    Attribute Inference Attack.
    
    Evaluates whether an attacker can infer sensitive attributes
    (e.g., race, gender, age) from Z-view.
    
    attack_success = AUC (lower is better for privacy)
    """
    
    attack_type = AttackType.ATTRIBUTE_INFERENCE
    
    def __init__(self, device: str = None, classifier=None, attribute_name: str = "unknown"):
        """
        Initialize attribute inference attack.
        
        Args:
            device: Compute device
            classifier: Attribute classifier model
            attribute_name: Name of target attribute
        """
        super().__init__(device)
        self.classifier = classifier
        self.attribute_name = attribute_name
        self.trained_classifier = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train the attribute classifier.
        
        Args:
            ctx: Attack training context
            train_images: Training images (Z-view)
            train_labels: Attribute labels
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
        Evaluate attribute inference attack.
        
        Args:
            ctx: Evaluation context
            test_images: Test images (Z-view)
            test_labels: True attribute labels
            
        Returns:
            AttackResult with AUC as attack_success
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
        """Train a simple classifier."""
        # Extract features
        features = self._extract_features(images)
        
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        
        # Store for k-NN classification
        self.train_features = features
        self.train_labels = labels
    
    def _extract_features(self, images) -> np.ndarray:
        """Extract features from images."""
        if self.classifier is not None:
            import torch
            with torch.no_grad():
                if hasattr(images, 'to'):
                    images = images.to(self.device)
                features = self.classifier.extract_features(images)
                if hasattr(features, 'cpu'):
                    features = features.cpu().numpy()
            return features
        
        # Fallback: flatten images
        if hasattr(images, 'numpy'):
            images = images.numpy()
        return images.reshape(len(images), -1)
    
    def _compute_metrics(self, images, labels) -> AttributeInferenceMetrics:
        """Compute attribute inference metrics."""
        features = self._extract_features(images)
        
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        
        # Predict using k-NN or classifier
        if hasattr(self, 'train_features') and self.train_features is not None:
            predictions, scores = self._knn_predict(features)
        else:
            # Random predictions
            num_classes = len(np.unique(labels))
            predictions = np.random.randint(0, num_classes, len(labels))
            scores = np.random.rand(len(labels), num_classes)
        
        # Compute accuracy
        accuracy = np.mean(predictions == labels)
        
        # Compute AUC (one-vs-rest for multi-class)
        auc = self._compute_multiclass_auc(scores, labels)
        
        return AttributeInferenceMetrics(
            auc=auc,
            accuracy=accuracy,
            attribute_name=self.attribute_name,
            num_classes=len(np.unique(labels)),
            num_samples=len(labels),
        )
    
    def _knn_predict(self, features: np.ndarray, k: int = 5):
        """K-NN prediction."""
        predictions = []
        scores = []
        
        num_classes = len(np.unique(self.train_labels))
        
        for feat in features:
            # Compute distances
            dists = np.linalg.norm(self.train_features - feat, axis=1)
            
            # Get k nearest neighbors
            k_idx = np.argsort(dists)[:k]
            k_labels = self.train_labels[k_idx]
            
            # Vote
            pred = np.bincount(k_labels.astype(int), minlength=num_classes).argmax()
            predictions.append(pred)
            
            # Scores (inverse distance weighted)
            score = np.zeros(num_classes)
            for idx, label in zip(k_idx, k_labels):
                score[int(label)] += 1.0 / (dists[idx] + 1e-8)
            score /= score.sum() + 1e-8
            scores.append(score)
        
        return np.array(predictions), np.array(scores)
    
    def _compute_multiclass_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute macro-averaged AUC for multi-class."""
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
        """Compute binary AUC."""
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
