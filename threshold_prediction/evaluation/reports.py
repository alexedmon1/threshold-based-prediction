"""
HTML report generation for threshold-based prediction results.

Creates comprehensive HTML reports with embedded visualizations,
summary statistics, and key threshold identification.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from threshold_prediction.evaluation.metrics import ResultsEvaluator
from threshold_prediction.evaluation.visualization import ResultsVisualizer


class InflectionPointDetector:
    """Detect key thresholds where accuracy/recall increases significantly."""

    @staticmethod
    def find_inflection_points(
        x: np.ndarray,
        y: np.ndarray,
        method: str = "derivative"
    ) -> List[int]:
        """
        Find inflection points in accuracy/recall curve.

        Args:
            x: Threshold values
            y: Metric values (accuracy, recall, etc.)
            method: Detection method ('derivative', 'curvature', or 'peaks')

        Returns:
            List of indices where inflection points occur
        """
        if len(x) < 3:
            return []

        if method == "derivative":
            # Find points where first derivative changes significantly
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)

            # Identify sign changes in second derivative
            sign_changes = np.where(np.diff(np.sign(d2y)))[0]

            # Filter significant changes (above threshold)
            threshold = np.std(d2y) * 0.5
            significant = [idx for idx in sign_changes if abs(d2y[idx]) > threshold]

            return significant

        elif method == "curvature":
            # Calculate curvature
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)

            curvature = np.abs(d2y) / (1 + dy**2)**(3/2)

            # Find peaks in curvature
            peaks, _ = signal.find_peaks(curvature, height=np.std(curvature) * 0.5)

            return peaks.tolist()

        elif method == "peaks":
            # Find peaks in the first derivative (steep increases)
            dy = np.gradient(y, x)

            peaks, _ = signal.find_peaks(dy, height=np.std(dy) * 0.5)

            return peaks.tolist()

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def identify_key_thresholds(
        metrics_df: pd.DataFrame,
        metric: str = "accuracy"
    ) -> Dict:
        """
        Identify key threshold points.

        Args:
            metrics_df: DataFrame with metrics
            metric: Metric to analyze

        Returns:
            Dictionary with key threshold information
        """
        x = metrics_df['threshold'].values
        y = metrics_df[metric].values

        # Find inflection points
        inflection_idx = InflectionPointDetector.find_inflection_points(x, y, method="derivative")
        peak_idx = InflectionPointDetector.find_inflection_points(x, y, method="peaks")

        # Combine and remove duplicates
        key_idx = sorted(list(set(inflection_idx + peak_idx)))

        key_thresholds = []
        for idx in key_idx:
            if 0 <= idx < len(x):
                key_thresholds.append({
                    'threshold': x[idx],
                    'value': y[idx],
                    'type': 'inflection'
                })

        # Add maximum
        max_idx = np.argmax(y)
        key_thresholds.append({
            'threshold': x[max_idx],
            'value': y[max_idx],
            'type': 'maximum'
        })

        return {
            'metric': metric,
            'key_thresholds': key_thresholds,
            'n_inflection_points': len(inflection_idx)
        }


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for analysis results."""

    def __init__(self, evaluator: ResultsEvaluator, visualizer: ResultsVisualizer):
        """
        Initialize HTML report generator.

        Args:
            evaluator: ResultsEvaluator with metrics
            visualizer: ResultsVisualizer for plots
        """
        self.evaluator = evaluator
        self.visualizer = visualizer

    def figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for HTML embedding.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded string
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return image_base64

    def generate_html_report(
        self,
        output_path: Path,
        target_variable: str = "exposure",
        include_inflection_points: bool = True
    ):
        """
        Generate comprehensive HTML report.

        Args:
            output_path: Path to save HTML file
            target_variable: Name of target variable
            include_inflection_points: Include inflection point analysis
        """
        output_path = Path(output_path)

        # Calculate metrics if needed
        if self.evaluator.metrics is None:
            self.evaluator.calculate_all_metrics()

        # Calculate ROC if needed
        if self.evaluator.roc_data is None:
            self.evaluator.calculate_roc_analysis()

        # Get summary statistics
        summary = self.evaluator.summary_statistics()
        best = self.evaluator.get_best_threshold('accuracy')

        # Detect key thresholds
        key_thresholds = None
        if include_inflection_points:
            key_thresholds = InflectionPointDetector.identify_key_thresholds(
                self.evaluator.metrics,
                metric='accuracy'
            )

        # Generate plots
        figures = self._generate_all_figures(target_variable, key_thresholds)

        # Build HTML
        html = self._build_html(
            summary=summary,
            best=best,
            figures=figures,
            key_thresholds=key_thresholds,
            target_variable=target_variable
        )

        # Save HTML
        with open(output_path, 'w') as f:
            f.write(html)

        # Close all figures
        plt.close('all')

        print(f"HTML report saved to {output_path}")

    def _generate_all_figures(
        self,
        target_variable: str,
        key_thresholds: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate all figures and convert to base64."""
        figures = {}

        # 1. Accuracy vs threshold (with key thresholds marked)
        fig = self.visualizer.plot_accuracy_vs_threshold(
            self.evaluator.metrics,
            title=f"Accuracy vs {target_variable.replace('_', ' ').title()} Threshold"
        )

        # Add key threshold markers if available
        if key_thresholds:
            ax = fig.gca()
            for kt in key_thresholds['key_thresholds']:
                if kt['type'] == 'inflection':
                    ax.axvline(
                        kt['threshold'],
                        color='orange',
                        linestyle='--',
                        alpha=0.5,
                        label='Key Threshold' if kt == key_thresholds['key_thresholds'][0] else ''
                    )

        figures['accuracy'] = self.figure_to_base64(fig)
        plt.close(fig)

        # 2. Metrics comparison
        fig = self.visualizer.plot_metrics_comparison(
            self.evaluator.metrics,
            title=f"Classification Metrics - {target_variable.replace('_', ' ').title()}"
        )
        figures['metrics_comparison'] = self.figure_to_base64(fig)
        plt.close(fig)

        # 3. ROC curve
        fig = self.visualizer.plot_roc_curve(
            self.evaluator,
            title=f"ROC Curve - {target_variable.replace('_', ' ').title()}"
        )
        figures['roc_curve'] = self.figure_to_base64(fig)
        plt.close(fig)

        # 4. Group distributions
        fig = self.visualizer.plot_group_distributions(
            self.evaluator.metrics,
            title=f"Group Sizes - {target_variable.replace('_', ' ').title()}"
        )
        figures['groups'] = self.figure_to_base64(fig)
        plt.close(fig)

        # 5. Confusion matrix for optimal threshold
        best = self.evaluator.get_best_threshold('accuracy')
        if best['threshold'] in self.evaluator.results:
            result = self.evaluator.results[best['threshold']]
            if result.get('predictions') is not None:
                fig = self.visualizer.plot_confusion_matrix(
                    result['true_labels'],
                    result['predictions'],
                    best['threshold']
                )
                figures['confusion_matrix'] = self.figure_to_base64(fig)
                plt.close(fig)

        return figures

    def _build_html(
        self,
        summary: Dict,
        best: Dict,
        figures: Dict[str, str],
        key_thresholds: Optional[Dict],
        target_variable: str
    ) -> str:
        """Build complete HTML document."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Threshold-Based Prediction Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .figure {{
            text-align: center;
            margin: 30px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .figure-caption {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Threshold-Based Prediction Analysis Report</h1>
        <p>Target Variable: {target_variable.replace('_', ' ').title()}</p>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Optimal Threshold</div>
                <div class="stat-value">{best['threshold']:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Accuracy</div>
                <div class="stat-value">{best['accuracy']:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ROC AUC</div>
                <div class="stat-value">{summary.get('auc', 0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Thresholds Tested</div>
                <div class="stat-value">{summary['n_thresholds']}</div>
            </div>
        </div>

        <div class="highlight">
            <strong>Recommended Threshold:</strong> {best['threshold']:.4f}<br>
            This threshold yields the highest classification accuracy ({best['accuracy']:.1%})
            with precision of {best['precision']:.1%} and recall of {best['recall']:.1%}.
        </div>
    </div>
"""

        # Key Thresholds Section
        if key_thresholds:
            html += """
    <div class="section">
        <h2>Key Thresholds Analysis</h2>
        <p>These thresholds represent points where classification accuracy shows significant changes (inflection points).</p>
        <table>
            <thead>
                <tr>
                    <th>Threshold</th>
                    <th>Accuracy</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
"""
            for kt in key_thresholds['key_thresholds']:
                html += f"""
                <tr>
                    <td>{kt['threshold']:.4f}</td>
                    <td>{kt['value']:.1%}</td>
                    <td>{kt['type'].title()}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
    </div>
"""

        # Detailed Metrics Section
        html += f"""
    <div class="section">
        <h2>Detailed Performance Metrics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Mean Accuracy</div>
                <div class="stat-value">{summary['accuracy_mean']:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Accuracy Std Dev</div>
                <div class="stat-value">{summary['accuracy_std']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best F1 Score</div>
                <div class="stat-value">{summary['f1_max']:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Precision</div>
                <div class="stat-value">{summary['precision_mean']:.1%}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Visualizations</h2>

        <div class="figure">
            <img src="data:image/png;base64,{figures['accuracy']}" alt="Accuracy vs Threshold">
            <div class="figure-caption">Figure 1: Classification accuracy across threshold values. Orange dashed lines indicate key inflection points.</div>
        </div>

        <div class="figure">
            <img src="data:image/png;base64,{figures['metrics_comparison']}" alt="Metrics Comparison">
            <div class="figure-caption">Figure 2: Comparison of classification metrics (accuracy, precision, recall, F1) across thresholds.</div>
        </div>

        <div class="figure">
            <img src="data:image/png;base64,{figures['roc_curve']}" alt="ROC Curve">
            <div class="figure-caption">Figure 3: ROC curve showing classifier performance. The red star indicates the optimal threshold.</div>
        </div>

        <div class="figure">
            <img src="data:image/png;base64,{figures['groups']}" alt="Group Distributions">
            <div class="figure-caption">Figure 4: Distribution of subjects in low vs high exposure groups across thresholds.</div>
        </div>
"""

        if 'confusion_matrix' in figures:
            html += f"""
        <div class="figure">
            <img src="data:image/png;base64,{figures['confusion_matrix']}" alt="Confusion Matrix">
            <div class="figure-caption">Figure 5: Confusion matrix for the optimal threshold.</div>
        </div>
"""

        html += """
    </div>

    <div class="footer">
        <p>Generated by Threshold-Based Prediction Package</p>
        <p>For more information, see the package documentation</p>
    </div>
</body>
</html>
"""

        return html
