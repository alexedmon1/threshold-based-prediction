"""
Visualization tools for threshold-based prediction results.

Provides plotting functions for accuracy curves, ROC curves, confusion matrices,
and PCA visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from threshold_prediction.evaluation.metrics import ResultsEvaluator


class ResultsVisualizer:
    """Visualize threshold-based prediction results."""

    def __init__(self, results: Optional[Dict] = None, style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.

        Args:
            results: Dictionary of threshold scan results
            style: Matplotlib style to use
        """
        self.results = results
        self.style = style

        # Set style
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

        # Color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
        }

    def plot_accuracy_vs_threshold(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Accuracy vs Threshold",
        save_path: Optional[Path] = None,
        show_ci: bool = True
    ) -> Figure:
        """
        Plot accuracy across thresholds.

        Args:
            metrics_df: DataFrame with threshold metrics
            title: Plot title
            save_path: Optional path to save figure
            show_ci: Show confidence intervals

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Main accuracy line
        ax.plot(
            metrics_df['threshold'],
            metrics_df['accuracy'],
            color=self.colors['primary'],
            linewidth=2,
            label='Accuracy'
        )

        # Confidence intervals
        if show_ci and 'accuracy_ci_lower' in metrics_df.columns:
            ax.fill_between(
                metrics_df['threshold'],
                metrics_df['accuracy_ci_lower'],
                metrics_df['accuracy_ci_upper'],
                alpha=0.2,
                color=self.colors['primary'],
                label='95% CI'
            )

        # Mark optimal threshold
        max_acc_idx = metrics_df['accuracy'].idxmax()
        optimal_thresh = metrics_df.loc[max_acc_idx, 'threshold']
        optimal_acc = metrics_df.loc[max_acc_idx, 'accuracy']

        ax.plot(
            optimal_thresh,
            optimal_acc,
            'r*',
            markersize=15,
            label=f'Optimal: {optimal_thresh:.3f}'
        )

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
        title: str = "Classification Metrics vs Threshold",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot multiple metrics on same axes.

        Args:
            metrics_df: DataFrame with threshold metrics
            metrics: List of metrics to plot
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['success'],
            self.colors['warning']
        ]

        for i, metric in enumerate(metrics):
            if metric in metrics_df.columns:
                ax.plot(
                    metrics_df['threshold'],
                    metrics_df[metric],
                    label=metric.replace('_', ' ').title(),
                    linewidth=2,
                    color=colors[i % len(colors)]
                )

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curve(
        self,
        evaluator: ResultsEvaluator,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot ROC curve.

        Args:
            evaluator: ResultsEvaluator with ROC data
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure object
        """
        if evaluator.roc_data is None:
            evaluator.calculate_roc_analysis()

        roc_data = evaluator.roc_data

        fig, ax = plt.subplots(figsize=(8, 8))

        # ROC curve
        ax.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            color=self.colors['primary'],
            linewidth=2,
            label=f"AUC = {roc_data['auc']:.3f}"
        )

        # Diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        # Mark optimal point
        ax.plot(
            roc_data['optimal_fpr'],
            roc_data['optimal_tpr'],
            'r*',
            markersize=15,
            label=f"Optimal (thresh={roc_data['optimal_threshold']:.3f})"
        )

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        title: Optional[str] = None,
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            threshold: Threshold value
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure object
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        if title is None:
            title = f'Confusion Matrix (Threshold = {threshold:.3f})'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Low', 'High'])
        ax.set_yticklabels(['Low', 'High'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_group_distributions(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Group Size Distribution",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot distribution of samples in low vs high groups.

        Args:
            metrics_df: DataFrame with metrics
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            metrics_df['threshold'],
            metrics_df['n_low'],
            label='Low Exposure Group',
            linewidth=2,
            color=self.colors['success']
        )

        ax.plot(
            metrics_df['threshold'],
            metrics_df['n_high'],
            label='High Exposure Group',
            linewidth=2,
            color=self.colors['danger']
        )

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Number of Subjects', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pca_variance(
        self,
        pca: PCA,
        title: str = "PCA Explained Variance",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot PCA explained variance.

        Args:
            pca: Fitted PCA object
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            color=self.colors['primary'],
            alpha=0.7
        )
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Individual Component Variance', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Cumulative variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(
            range(1, len(cumsum) + 1),
            cumsum,
            marker='o',
            color=self.colors['primary'],
            linewidth=2
        )
        ax2.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Variance', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_summary_report(
        self,
        evaluator: ResultsEvaluator,
        output_dir: Path,
        target_variable: str = "exposure"
    ):
        """
        Create comprehensive visualization report.

        Args:
            evaluator: ResultsEvaluator with calculated metrics
            output_dir: Directory to save plots
            target_variable: Name of target variable for titles
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating visualization report...")

        # Calculate metrics if needed
        if evaluator.metrics is None:
            evaluator.calculate_all_metrics()

        # 1. Accuracy vs threshold
        print("  - Accuracy vs threshold plot")
        self.plot_accuracy_vs_threshold(
            evaluator.metrics,
            title=f"Accuracy vs {target_variable.replace('_', ' ').title()} Threshold",
            save_path=output_dir / "accuracy_vs_threshold.png"
        )
        plt.close()

        # 2. Metrics comparison
        print("  - Metrics comparison plot")
        self.plot_metrics_comparison(
            evaluator.metrics,
            title=f"Classification Metrics - {target_variable.replace('_', ' ').title()}",
            save_path=output_dir / "metrics_comparison.png"
        )
        plt.close()

        # 3. ROC curve
        print("  - ROC curve")
        self.plot_roc_curve(
            evaluator,
            title=f"ROC Curve - {target_variable.replace('_', ' ').title()}",
            save_path=output_dir / "roc_curve.png"
        )
        plt.close()

        # 4. Group distributions
        print("  - Group size distribution")
        self.plot_group_distributions(
            evaluator.metrics,
            title=f"Group Sizes - {target_variable.replace('_', ' ').title()}",
            save_path=output_dir / "group_distributions.png"
        )
        plt.close()

        # 5. Confusion matrix for optimal threshold
        print("  - Confusion matrix (optimal threshold)")
        best = evaluator.get_best_threshold('accuracy')
        if best['threshold'] in evaluator.results:
            result = evaluator.results[best['threshold']]
            if result.get('predictions') is not None:
                self.plot_confusion_matrix(
                    result['true_labels'],
                    result['predictions'],
                    best['threshold'],
                    save_path=output_dir / "confusion_matrix_optimal.png"
                )
                plt.close()

        print(f"Visualization report saved to {output_dir}")

    @staticmethod
    def close_all():
        """Close all open figures."""
        plt.close('all')
