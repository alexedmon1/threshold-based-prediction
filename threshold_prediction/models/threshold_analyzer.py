"""
Threshold-based SVM analyzer for toxicological exposure prediction.

Core algorithm for identifying exposure thresholds using brain imaging biomarkers
and support vector machine classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from threshold_prediction.utils.config import ModelConfig, ThresholdConfig


class ThresholdAnalyzer:
    """
    Threshold-based SVM analyzer.

    Identifies toxicological thresholds by scanning exposure levels and finding
    the threshold where brain imaging patterns best separate into two groups.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize threshold analyzer.

        Args:
            config: ModelConfig object with analysis parameters
        """
        self.config = config or ModelConfig()
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load prepared data for analysis.

        Args:
            data_path: Path to CSV file with imaging data and metadata

        Returns:
            Loaded DataFrame
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = pd.read_csv(data_path, index_col=0)

        return self.data

    def create_binary_groups(
        self,
        target_variable: str,
        threshold: float
    ) -> Tuple[pd.Series, Dict[str, int]]:
        """
        Create binary groups based on threshold.

        Args:
            target_variable: Column name of exposure variable
            threshold: Threshold value to split groups

        Returns:
            Tuple of (binary labels Series, group counts dictionary)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if target_variable not in self.data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")

        # Create binary labels
        labels = (self.data[target_variable] > threshold).astype(int)
        labels.name = "group"

        # Count groups
        counts = {
            "low": (labels == 0).sum(),
            "high": (labels == 1).sum()
        }

        return labels, counts

    def preprocess_features(
        self,
        features: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Preprocess features: impute missing values and standardize.

        Args:
            features: Feature DataFrame
            fit: If True, fit imputer and scaler; if False, use existing

        Returns:
            Preprocessed feature array
        """
        if fit:
            # Initialize and fit imputer
            self.imputer = SimpleImputer(strategy='mean')
            features_imputed = self.imputer.fit_transform(features)

            # Initialize and fit scaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_imputed)
        else:
            # Use existing imputer and scaler
            if self.imputer is None or self.scaler is None:
                raise ValueError("Imputer/scaler not fitted. Set fit=True first.")

            features_imputed = self.imputer.transform(features)
            features_scaled = self.scaler.transform(features_imputed)

        return features_scaled

    def apply_pca(
        self,
        features: np.ndarray,
        variance_threshold: float = 0.90
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.

        Args:
            features: Standardized feature array
            variance_threshold: Fraction of variance to retain (default: 0.90)

        Returns:
            Tuple of (transformed features, fitted PCA object)
        """
        # Initial fit to determine number of components
        pca_init = PCA()
        pca_init.fit(features)

        # Find number of components for desired variance
        cumsum = np.cumsum(pca_init.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1

        # Fit final PCA with selected components
        pca = PCA(n_components=n_components, random_state=self.config.random_state)
        features_pca = pca.fit_transform(features)

        return features_pca, pca

    def train_svm_cv(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_folds: Optional[int] = None
    ) -> Dict:
        """
        Train SVM with cross-validation.

        Args:
            features: Feature array (after PCA)
            labels: Binary group labels
            n_folds: Number of CV folds (None for leave-one-out)

        Returns:
            Dictionary with predictions and true labels
        """
        if n_folds is None or n_folds >= len(features):
            # Leave-one-out cross-validation
            cv = LeaveOneOut()
            n_splits = len(features)
        else:
            # K-fold cross-validation
            cv = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            n_splits = n_folds

        predictions = []
        true_labels = []

        for train_idx, test_idx in cv.split(features):
            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Train SVM
            clf = LinearSVC(
                class_weight='balanced' if self.config.class_weight == 'balanced' else None,
                random_state=self.config.random_state,
                max_iter=10000
            )
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)

            predictions.extend(y_pred)
            true_labels.extend(y_test)

        return {
            'predictions': np.array(predictions),
            'true_labels': np.array(true_labels),
            'n_splits': n_splits
        }

    def calculate_accuracy(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """
        Calculate classification accuracy.

        Args:
            predictions: Predicted labels
            true_labels: True labels

        Returns:
            Accuracy (fraction correct)
        """
        correct = (predictions == true_labels).sum()
        total = len(predictions)
        return correct / total if total > 0 else 0.0

    def analyze_threshold(
        self,
        target_variable: str,
        threshold: float,
        feature_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze a single threshold value.

        Args:
            target_variable: Exposure variable column name
            threshold: Threshold value to test
            feature_columns: List of feature columns (None = all numeric except target)

        Returns:
            Dictionary with analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Create binary groups
        labels, counts = self.create_binary_groups(target_variable, threshold)

        # Check for minimum samples per group
        if counts['low'] < 2 or counts['high'] < 2:
            return {
                'threshold': threshold,
                'counts': counts,
                'accuracy': None,
                'n_components': None,
                'error': 'Insufficient samples in one or both groups'
            }

        # Select features
        if feature_columns is None:
            # Use all numeric columns except target and known metadata
            exclude_cols = {target_variable, 'group', 'study_group', 'age', 'sex',
                           'treatment_group', 'dose', 'subject_id'}
            feature_columns = [
                col for col in self.data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.data[col])
            ]

        features_df = self.data[feature_columns]

        # Preprocess
        features_scaled = self.preprocess_features(features_df, fit=True)

        # Apply PCA
        features_pca, pca = self.apply_pca(features_scaled, self.config.pca_variance)

        # Train and evaluate
        cv_results = self.train_svm_cv(
            features_pca,
            labels.values,
            n_folds=self.config.cv_folds if self.config.cv_method == 'kfold' else None
        )

        # Calculate accuracy
        accuracy = self.calculate_accuracy(
            cv_results['predictions'],
            cv_results['true_labels']
        )

        return {
            'threshold': threshold,
            'counts': counts,
            'accuracy': accuracy,
            'n_components': pca.n_components_,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'predictions': cv_results['predictions'],
            'true_labels': cv_results['true_labels'],
            'pca': pca
        }

    def scan_thresholds(
        self,
        target_variable: str,
        threshold_range: Tuple[float, float],
        threshold_step: float,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scan multiple threshold values.

        Args:
            target_variable: Exposure variable column name
            threshold_range: (min, max) threshold values
            threshold_step: Step size for scanning
            feature_columns: List of feature columns to use

        Returns:
            DataFrame with results for each threshold
        """
        # Generate threshold values
        thresholds = np.arange(
            threshold_range[0],
            threshold_range[1] + threshold_step,
            threshold_step
        )

        results = []

        print(f"Scanning {len(thresholds)} thresholds from {threshold_range[0]} to {threshold_range[1]}...")

        for thresh in tqdm(thresholds, desc="Threshold scan"):
            result = self.analyze_threshold(
                target_variable=target_variable,
                threshold=thresh,
                feature_columns=feature_columns
            )

            # Store simplified results
            results.append({
                'threshold': result['threshold'],
                'accuracy': result['accuracy'],
                'n_low': result['counts']['low'],
                'n_high': result['counts']['high'],
                'n_components': result['n_components'],
                'explained_variance': result.get('explained_variance')
            })

            # Store full results
            self.results[thresh] = result

        df_results = pd.DataFrame(results)

        return df_results

    def get_optimal_threshold(self) -> Dict:
        """
        Get the threshold with highest accuracy.

        Returns:
            Dictionary with optimal threshold information
        """
        if not self.results:
            raise ValueError("No results available. Run scan_thresholds() first.")

        # Find threshold with max accuracy
        max_accuracy = 0
        optimal_threshold = None

        for thresh, result in self.results.items():
            if result.get('accuracy') is not None and result['accuracy'] > max_accuracy:
                max_accuracy = result['accuracy']
                optimal_threshold = thresh

        if optimal_threshold is None:
            raise ValueError("No valid results found")

        return {
            'threshold': optimal_threshold,
            'accuracy': max_accuracy,
            'result': self.results[optimal_threshold]
        }

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'ThresholdAnalyzer':
        """
        Create analyzer from configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configured ThresholdAnalyzer instance
        """
        from threshold_prediction.utils.config import ConfigManager

        config = ConfigManager.load_threshold_config(config_path)

        return cls(config=config.model)

    def save_results(self, output_path: Union[str, Path]):
        """
        Save analysis results to CSV.

        Args:
            output_path: Path to save results
        """
        if not self.results:
            raise ValueError("No results to save")

        results_list = []
        for thresh, result in self.results.items():
            results_list.append({
                'threshold': thresh,
                'accuracy': result.get('accuracy'),
                'n_low': result['counts']['low'],
                'n_high': result['counts']['high'],
                'n_components': result.get('n_components'),
                'explained_variance': result.get('explained_variance')
            })

        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)

        print(f"Results saved to {output_path}")
