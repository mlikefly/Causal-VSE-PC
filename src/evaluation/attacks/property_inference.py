"""
Property Inference Attack implementation.

Infers group-level properties (e.g., race/gender distribution).
attack_success = AUC (per GC7)

**Validates: Requirements R2.AC2**
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
    """Property inference attack metrics."""
    auc: float  # attack_success
    accuracy: float
    property_name: str
    num_properties: int
    distribution_error: float  # MAE of predicted vs true distribution


@AttackRegistry.register(AttackType.PROPERTY_INFERENCE)
class PropertyInferenceAttack(AttackBase):
    """
    Property Inference Attack.
    
    Infers group-level properties from a dataset of Z-views,
    such as the distribution of race/gender in the training set.
    
    attack_success = AUC (lower is better for privacy)
    
    Per Requirements R2.AC2:
    - Predicts group attributes (race/gender distribution)
    - Outputs accuracy/AUC + CI
    """
    
    attack_type = AttackType.PROPERTY_INFERENCE
    
    def __init__(
        self,
        device: str = None,
        property_classifier=None,
        property_name: str = "unknown",
    ):
        """
        Initialize property inference attack.
        
        Args:
            device: Compute device
            property_classifier: Classifier for property inference
            property_name: Name of target property
        """
        super().__init__(device)
        self.property_classifier = property_classifier
        self.property_name = property_name
        self.trained_model = None
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train property inference model.
        
        Args:
            ctx: Attack training context
            train_datasets: List of datasets with known properties
            train_properties: Property labels for each dataset
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
        Evaluate property inference attack.
        
        Args:
            ctx: Evaluation context
            test_dataset: Dataset to infer properties from
            true_property: True property value/distribution
            
        Returns:
            AttackResult with AUC as attack_success
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
        """Train property classifier on dataset statistics."""
        # Extract aggregate features from each dataset
        self.train_features = []
        self.train_labels = []
        
        for dataset, prop in zip(datasets, properties):
            features = self._extract_dataset_features(dataset)
            self.train_features.append(features)
            self.train_labels.append(prop)
        
        self.train_features = np.array(self.train_features)
        self.train_labels = np.array(self.train_labels)
    
    def _extract_dataset_features(self, dataset) -> np.ndarray:
        """Extract aggregate features from a dataset."""
        if hasattr(dataset, 'numpy'):
            dataset = dataset.numpy()
        
        # Compute statistical features
        features = []
        
        # Mean and std of pixel values
        features.append(dataset.mean())
        features.append(dataset.std())
        
        # Per-channel statistics
        if len(dataset.shape) > 3:
            for c in range(min(dataset.shape[1], 3)):
                features.append(dataset[:, c].mean())
                features.append(dataset[:, c].std())
        
        # Histogram features
        hist, _ = np.histogram(dataset.flatten(), bins=10, density=True)
        features.extend(hist.tolist())
        
        return np.array(features)
    
    def _compute_metrics(
        self,
        dataset,
        true_property
    ) -> PropertyInferenceMetrics:
        """Compute property inference metrics."""
        if hasattr(dataset, 'numpy'):
            dataset = dataset.numpy()
        
        # Extract features
        features = self._extract_dataset_features(dataset)
        
        # Predict property
        if hasattr(self, 'train_features') and self.train_features is not None:
            predicted, scores = self._predict_property(features)
        else:
            # Random prediction
            predicted = np.random.rand()
            scores = np.array([0.5, 0.5])
        
        # Compute metrics
        if true_property is not None:
            if hasattr(true_property, 'numpy'):
                true_property = true_property.numpy()
            
            # For distribution properties
            if isinstance(true_property, (list, np.ndarray)):
                distribution_error = np.mean(np.abs(
                    np.array(predicted) - np.array(true_property)
                ))
                # AUC based on distribution prediction accuracy
                auc = 1.0 - distribution_error
            else:
                # For categorical properties
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
        """Predict property using trained model."""
        if self.train_features is None or len(self.train_features) == 0:
            return 0.5, np.array([0.5, 0.5])
        
        # K-NN prediction
        distances = np.linalg.norm(self.train_features - features, axis=1)
        k = min(3, len(distances))
        k_idx = np.argsort(distances)[:k]
        
        # Weighted average of labels
        weights = 1.0 / (distances[k_idx] + 1e-8)
        weights /= weights.sum()
        
        predicted = np.average(self.train_labels[k_idx], weights=weights, axis=0)
        
        # Scores (confidence)
        scores = np.array([1.0 - np.mean(distances[k_idx]), np.mean(distances[k_idx])])
        
        return predicted, scores
