"""
Ensemble methods for combining predictions across statistical measures.

Implements bagging and majority voting to combine SVM predictions from
different statistical features (median, variance, skewness, etc.).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from collections import Counter

from threshold_prediction.models.threshold_analyzer import ThresholdAnalyzer


class EnsembleBagging:
    """
    Ensemble bagging for combining predictions across statistical measures.

    Combines predictions from multiple ThresholdAnalyzer models trained on
    different statistical summaries of the imaging data.
    """

    def __init__(self, voting: str = "majority"):
        """
        Initialize ensemble bagging.

        Args:
            voting: Voting method ('majority' or 'weighted')
        """
        if voting not in ["majority", "weighted"]:
            raise ValueError("voting must be 'majority' or 'weighted'")

        self.voting = voting
        self.analyzers: Dict[str, ThresholdAnalyzer] = {}
        self.ensemble_results: Dict = {}

    def add_analyzer(self, name: str, analyzer: ThresholdAnalyzer):
        """
        Add an analyzer to the ensemble.

        Args:
            name: Name for this analyzer (e.g., 'median', 'variance')
            analyzer: Fitted ThresholdAnalyzer instance
        """
        self.analyzers[name] = analyzer

    def majority_vote(
        self,
        predictions_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine predictions using majority voting.

        Args:
            predictions_dict: Dictionary mapping analyzer names to prediction arrays

        Returns:
            Combined predictions array
        """
        # Stack predictions
        all_predictions = np.vstack(list(predictions_dict.values()))

        # Majority vote for each sample
        n_samples = all_predictions.shape[1]
        ensemble_predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            votes = all_predictions[:, i]
            # Count votes
            counter = Counter(votes)
            # Get most common vote
            ensemble_predictions[i] = counter.most_common(1)[0][0]

        return ensemble_predictions

    def weighted_vote(
        self,
        predictions_dict: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Combine predictions using weighted voting.

        Args:
            predictions_dict: Dictionary mapping analyzer names to prediction arrays
            weights: Optional dictionary of weights per analyzer (default: equal weights)

        Returns:
            Combined predictions array
        """
        if weights is None:
            # Equal weights
            weights = {name: 1.0 for name in predictions_dict.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Stack predictions
        all_predictions = np.vstack(list(predictions_dict.values()))
        weight_array = np.array([normalized_weights[name] for name in predictions_dict.keys()])

        # Weighted vote
        n_samples = all_predictions.shape[1]
        ensemble_predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            votes = all_predictions[:, i]
            # Calculate weighted votes for each class
            vote_0_weight = sum(weight_array[votes == 0])
            vote_1_weight = sum(weight_array[votes == 1])

            # Assign class based on higher weight
            ensemble_predictions[i] = 1 if vote_1_weight > vote_0_weight else 0

        return ensemble_predictions

    def combine_predictions(
        self,
        threshold: float,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Combine predictions at a specific threshold.

        Args:
            threshold: Threshold value to combine predictions for
            weights: Optional weights for weighted voting

        Returns:
            Dictionary with combined predictions and accuracy
        """
        # Collect predictions from all analyzers
        predictions_dict = {}
        true_labels = None

        for name, analyzer in self.analyzers.items():
            if threshold not in analyzer.results:
                raise ValueError(f"Threshold {threshold} not found in analyzer '{name}'")

            result = analyzer.results[threshold]

            if result.get('predictions') is None:
                continue

            predictions_dict[name] = result['predictions']

            # Get true labels (should be same across all analyzers)
            if true_labels is None:
                true_labels = result['true_labels']

        if len(predictions_dict) == 0:
            return {
                'threshold': threshold,
                'ensemble_predictions': None,
                'true_labels': None,
                'accuracy': None,
                'individual_accuracies': {},
                'error': 'No valid predictions at this threshold'
            }

        # Combine predictions
        if self.voting == "majority":
            ensemble_predictions = self.majority_vote(predictions_dict)
        else:  # weighted
            ensemble_predictions = self.weighted_vote(predictions_dict, weights)

        # Calculate ensemble accuracy
        ensemble_accuracy = (ensemble_predictions == true_labels).sum() / len(true_labels)

        # Calculate individual accuracies
        individual_accuracies = {}
        for name, preds in predictions_dict.items():
            individual_accuracies[name] = (preds == true_labels).sum() / len(true_labels)

        return {
            'threshold': threshold,
            'ensemble_predictions': ensemble_predictions,
            'true_labels': true_labels,
            'accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies,
            'n_analyzers': len(predictions_dict)
        }

    def scan_ensemble(
        self,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Calculate ensemble predictions across all thresholds.

        Args:
            thresholds: List of thresholds to process (None = all common thresholds)

        Returns:
            DataFrame with ensemble results
        """
        if thresholds is None:
            # Find common thresholds across all analyzers
            threshold_sets = [set(analyzer.results.keys()) for analyzer in self.analyzers.values()]
            thresholds = sorted(list(set.intersection(*threshold_sets)))

        results = []

        for thresh in thresholds:
            result = self.combine_predictions(thresh)

            results.append({
                'threshold': result['threshold'],
                'ensemble_accuracy': result.get('accuracy'),
                'n_analyzers': result.get('n_analyzers'),
                **{f'{name}_accuracy': acc for name, acc in result.get('individual_accuracies', {}).items()}
            })

            # Store full result
            self.ensemble_results[thresh] = result

        return pd.DataFrame(results)

    def get_optimal_threshold(self) -> Dict:
        """
        Get threshold with best ensemble accuracy.

        Returns:
            Dictionary with optimal threshold information
        """
        if not self.ensemble_results:
            raise ValueError("No ensemble results available. Run scan_ensemble() first.")

        max_accuracy = 0
        optimal_threshold = None

        for thresh, result in self.ensemble_results.items():
            if result.get('accuracy') is not None and result['accuracy'] > max_accuracy:
                max_accuracy = result['accuracy']
                optimal_threshold = thresh

        if optimal_threshold is None:
            raise ValueError("No valid ensemble results found")

        return {
            'threshold': optimal_threshold,
            'accuracy': max_accuracy,
            'result': self.ensemble_results[optimal_threshold]
        }

    def compare_to_individuals(self) -> pd.DataFrame:
        """
        Compare ensemble performance to individual analyzers.

        Returns:
            DataFrame comparing accuracies
        """
        comparisons = []

        for thresh, result in self.ensemble_results.items():
            if result.get('accuracy') is None:
                continue

            row = {
                'threshold': thresh,
                'ensemble': result['accuracy']
            }

            # Add individual accuracies
            for name, acc in result.get('individual_accuracies', {}).items():
                row[name] = acc

            # Calculate improvement over best individual
            individual_accs = list(result.get('individual_accuracies', {}).values())
            if individual_accs:
                best_individual = max(individual_accs)
                row['improvement'] = result['accuracy'] - best_individual

            comparisons.append(row)

        return pd.DataFrame(comparisons)

    def get_contribution_weights(self) -> Dict[str, float]:
        """
        Calculate contribution weights for each analyzer based on individual performance.

        Returns:
            Dictionary mapping analyzer names to performance-based weights
        """
        # Calculate average accuracy for each analyzer
        avg_accuracies = {}

        for name in self.analyzers.keys():
            accuracies = []
            for result in self.ensemble_results.values():
                if name in result.get('individual_accuracies', {}):
                    accuracies.append(result['individual_accuracies'][name])

            if accuracies:
                avg_accuracies[name] = np.mean(accuracies)
            else:
                avg_accuracies[name] = 0.0

        # Normalize to sum to 1
        total = sum(avg_accuracies.values())
        if total > 0:
            weights = {k: v / total for k, v in avg_accuracies.items()}
        else:
            # Equal weights if no accuracies
            weights = {k: 1.0 / len(avg_accuracies) for k in avg_accuracies.keys()}

        return weights


class StatisticalFeatureEnsemble:
    """
    Create and manage ensemble of analyzers for different statistical features.

    Handles creating separate datasets for each statistical measure
    (median, variance, skew, etc.) and training analyzers on each.
    """

    @staticmethod
    def extract_statistic_columns(
        df: pd.DataFrame,
        statistic: str
    ) -> List[str]:
        """
        Extract columns for a specific statistic.

        Args:
            df: DataFrame with columns like "region_median", "region_variance"
            statistic: Statistic name (e.g., "median", "variance", "skew")

        Returns:
            List of column names containing that statistic
        """
        stat_cols = [col for col in df.columns if f"_{statistic}" in col.lower()]
        return stat_cols

    @staticmethod
    def create_statistic_dataset(
        df: pd.DataFrame,
        statistic: str,
        metadata_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create dataset for a specific statistic.

        Args:
            df: Full DataFrame
            statistic: Statistic to extract
            metadata_columns: Columns to include as metadata (e.g., age, dose)

        Returns:
            DataFrame with only the specified statistic columns + metadata
        """
        stat_cols = StatisticalFeatureEnsemble.extract_statistic_columns(df, statistic)

        if len(stat_cols) == 0:
            raise ValueError(f"No columns found for statistic '{statistic}'")

        # Include metadata if specified
        if metadata_columns:
            cols_to_keep = stat_cols + [col for col in metadata_columns if col in df.columns]
        else:
            cols_to_keep = stat_cols

        # Create new dataframe
        df_stat = df[cols_to_keep].copy()

        # Rename columns to remove statistic suffix
        rename_dict = {col: col.replace(f"_{statistic}", "") for col in stat_cols}
        df_stat.rename(columns=rename_dict, inplace=True)

        return df_stat
