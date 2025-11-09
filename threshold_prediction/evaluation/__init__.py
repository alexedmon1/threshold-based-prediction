"""
Evaluation module for threshold-based prediction.

Provides metrics calculation, visualization tools, and HTML report generation.
"""

from threshold_prediction.evaluation.metrics import (
    ClassificationMetrics,
    MetricsCalculator,
    ResultsEvaluator,
)
from threshold_prediction.evaluation.reports import (
    HTMLReportGenerator,
    InflectionPointDetector,
)
from threshold_prediction.evaluation.visualization import ResultsVisualizer

__all__ = [
    "ClassificationMetrics",
    "MetricsCalculator",
    "ResultsEvaluator",
    "ResultsVisualizer",
    "HTMLReportGenerator",
    "InflectionPointDetector",
]
