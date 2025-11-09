"""
Evaluation metrics for threshold-based prediction.

Provides comprehensive evaluation including ROC analysis, precision/recall,
F1 scores, and confidence intervals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    tpr: float  # True positive rate (sensitivity)
    fpr: float  # False positive rate (1 - specificity)
    specificity: float
    n_samples: int
    n_low: int
    n_high: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'threshold': self.threshold,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'tp': self.true_positives,
            'tn': self.true_negatives,
            'fp': self.false_positives,
            'fn': self.false_negatives,
            'tpr': self.tpr,
            'fpr': self.fpr,
            'specificity': self.specificity,
            'n_samples': self.n_samples,
            'n_low': self.n_low,
            'n_high': self.n_high,
        }


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""

    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Calculate confusion matrix components.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (TP, TN, FP, FN)
        """
        cm = confusion_matrix(y_true, y_pred)

        # Handle binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Fallback for edge cases
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

        return int(tp), int(tn), int(fp), int(fn)

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float
    ) -> ClassificationMetrics:
        """
        Calculate all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            threshold: Threshold value used

        Returns:
            ClassificationMetrics object
        """
        # Confusion matrix
        tp, tn, fp, fn = MetricsCalculator.calculate_confusion_matrix(y_true, y_pred)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Handle zero-division cases
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Group sizes
        n_low = (y_true == 0).sum()
        n_high = (y_true == 1).sum()

        return ClassificationMetrics(
            threshold=threshold,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            tpr=tpr,
            fpr=fpr,
            specificity=specificity,
            n_samples=len(y_true),
            n_low=n_low,
            n_high=n_high,
        )

    @staticmethod
    def calculate_roc_curve(
        results: Dict[float, Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve from threshold scan results.

        Args:
            results: Dictionary mapping thresholds to result dictionaries

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        fpr_list = []
        tpr_list = []
        thresh_list = []

        for threshold, result in sorted(results.items()):
            if result.get('true_labels') is None or result.get('predictions') is None:
                continue

            metrics = MetricsCalculator.calculate_metrics(
                result['true_labels'],
                result['predictions'],
                threshold
            )

            fpr_list.append(metrics.fpr)
            tpr_list.append(metrics.tpr)
            thresh_list.append(threshold)

        return np.array(fpr_list), np.array(tpr_list), np.array(thresh_list)

    @staticmethod
    def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Calculate Area Under Curve using trapezoidal rule.

        Args:
            fpr: False positive rates
            tpr: True positive rates

        Returns:
            AUC value
        """
        # Sort by FPR
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr_sorted, fpr_sorted)

        return float(auc)

    @staticmethod
    def find_optimal_threshold_roc(
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Find optimal threshold using ROC curve (maximize TPR - FPR).

        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Threshold values

        Returns:
            Tuple of (optimal_threshold, optimal_fpr, optimal_tpr)
        """
        # Calculate Youden's J statistic (TPR - FPR)
        j_scores = tpr - fpr

        # Find maximum
        optimal_idx = np.argmax(j_scores)

        return (
            thresholds[optimal_idx],
            fpr[optimal_idx],
            tpr[optimal_idx]
        )

    @staticmethod
    def calculate_confidence_interval(
        accuracy: float,
        n_samples: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for accuracy using Wilson score interval.

        Args:
            accuracy: Observed accuracy
            n_samples: Number of samples
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n_samples == 0:
            return (0.0, 0.0)

        # Z-score for confidence level
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        # Wilson score interval
        denominator = 1 + z**2 / n_samples
        centre = (accuracy + z**2 / (2 * n_samples)) / denominator
        adjustment = (z / denominator) * np.sqrt(
            accuracy * (1 - accuracy) / n_samples + z**2 / (4 * n_samples**2)
        )

        lower = max(0.0, centre - adjustment)
        upper = min(1.0, centre + adjustment)

        return (lower, upper)


class ResultsEvaluator:
    """Evaluate threshold scan results comprehensively."""

    def __init__(self, results: Dict[float, Dict]):
        """
        Initialize evaluator.

        Args:
            results: Dictionary mapping thresholds to result dictionaries
        """
        self.results = results
        self.metrics: Optional[pd.DataFrame] = None
        self.roc_data: Optional[Dict] = None

    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for all thresholds.

        Returns:
            DataFrame with metrics for each threshold
        """
        metrics_list = []

        for threshold, result in sorted(self.results.items()):
            if result.get('true_labels') is None or result.get('predictions') is None:
                continue

            metrics = MetricsCalculator.calculate_metrics(
                result['true_labels'],
                result['predictions'],
                threshold
            )

            # Add confidence intervals
            ci_lower, ci_upper = MetricsCalculator.calculate_confidence_interval(
                metrics.accuracy,
                metrics.n_samples
            )

            metric_dict = metrics.to_dict()
            metric_dict['accuracy_ci_lower'] = ci_lower
            metric_dict['accuracy_ci_upper'] = ci_upper

            metrics_list.append(metric_dict)

        self.metrics = pd.DataFrame(metrics_list)

        return self.metrics

    def calculate_roc_analysis(self) -> Dict:
        """
        Perform ROC curve analysis.

        Returns:
            Dictionary with ROC curve data and AUC
        """
        fpr, tpr, thresholds = MetricsCalculator.calculate_roc_curve(self.results)

        auc = MetricsCalculator.calculate_auc(fpr, tpr)

        optimal_thresh, optimal_fpr, optimal_tpr = MetricsCalculator.find_optimal_threshold_roc(
            fpr, tpr, thresholds
        )

        self.roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc,
            'optimal_threshold': optimal_thresh,
            'optimal_fpr': optimal_fpr,
            'optimal_tpr': optimal_tpr,
        }

        return self.roc_data

    def get_best_threshold(self, metric: str = 'accuracy') -> Dict:
        """
        Find threshold with best performance on specified metric.

        Args:
            metric: Metric to optimize ('accuracy', 'f1_score', 'precision', 'recall')

        Returns:
            Dictionary with best threshold information
        """
        if self.metrics is None:
            self.calculate_all_metrics()

        if metric not in self.metrics.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(self.metrics.columns)}")

        best_idx = self.metrics[metric].idxmax()
        best_row = self.metrics.iloc[best_idx]

        return {
            'threshold': best_row['threshold'],
            'metric': metric,
            'value': best_row[metric],
            'accuracy': best_row['accuracy'],
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1_score': best_row['f1_score'],
        }

    def export_metrics(self, output_path: str):
        """
        Export metrics to CSV.

        Args:
            output_path: Path to save CSV
        """
        if self.metrics is None:
            self.calculate_all_metrics()

        self.metrics.to_csv(output_path, index=False)

        print(f"Metrics exported to {output_path}")

    def summary_statistics(self) -> Dict:
        """
        Calculate summary statistics across all thresholds.

        Returns:
            Dictionary with summary statistics
        """
        if self.metrics is None:
            self.calculate_all_metrics()

        summary = {
            'n_thresholds': len(self.metrics),
            'accuracy_mean': self.metrics['accuracy'].mean(),
            'accuracy_std': self.metrics['accuracy'].std(),
            'accuracy_max': self.metrics['accuracy'].max(),
            'accuracy_min': self.metrics['accuracy'].min(),
            'f1_mean': self.metrics['f1_score'].mean(),
            'f1_max': self.metrics['f1_score'].max(),
            'precision_mean': self.metrics['precision'].mean(),
            'recall_mean': self.metrics['recall'].mean(),
        }

        # Add ROC AUC if available
        if self.roc_data is not None:
            summary['auc'] = self.roc_data['auc']

        return summary
