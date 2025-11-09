"""
Data validation for neuroimaging datasets.

Provides quality checks for missing values, outliers, and data integrity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from threshold_prediction.utils.config import ValidationConfig


@dataclass
class ValidationReport:
    """Report of data validation results."""

    is_valid: bool
    n_subjects: int
    n_features: int
    missing_data: Dict[str, float]
    outliers: Dict[str, List]
    warnings: List[str]
    errors: List[str]

    def __str__(self) -> str:
        """Format validation report as string."""
        lines = [
            "=" * 60,
            "Data Validation Report",
            "=" * 60,
            f"Status: {'PASS' if self.is_valid else 'FAIL'}",
            f"Subjects: {self.n_subjects}",
            f"Features: {self.n_features}",
            "",
        ]

        # Missing data
        if self.missing_data:
            lines.append("Missing Data:")
            for col, pct in sorted(self.missing_data.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {col}: {pct:.1f}%")
            if len(self.missing_data) > 10:
                lines.append(f"  ... and {len(self.missing_data) - 10} more columns")
            lines.append("")

        # Outliers
        if self.outliers:
            lines.append(f"Outliers detected in {len(self.outliers)} columns")
            for col, outlier_indices in list(self.outliers.items())[:5]:
                lines.append(f"  {col}: {len(outlier_indices)} outliers")
            if len(self.outliers) > 5:
                lines.append(f"  ... and {len(self.outliers) - 5} more columns")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        # Errors
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class DataValidator:
    """Validator for neuroimaging datasets."""

    def __init__(self, config: ValidationConfig):
        """
        Initialize data validator.

        Args:
            config: ValidationConfig with validation parameters
        """
        self.config = config

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate a dataset.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationReport with results
        """
        warnings = []
        errors = []
        missing_data = {}
        outliers = {}

        # Basic checks
        n_subjects = len(df)
        n_features = len(df.columns)

        if n_subjects == 0:
            errors.append("Dataset is empty (0 subjects)")

        if n_features == 0:
            errors.append("Dataset has no features")

        # Check for missing data
        if self.config.check_missing:
            for col in df.columns:
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    pct_missing = (n_missing / n_subjects) * 100
                    missing_data[col] = pct_missing

                    if pct_missing > self.config.missing_threshold * 100:
                        errors.append(
                            f"Column '{col}' has {pct_missing:.1f}% missing data "
                            f"(threshold: {self.config.missing_threshold * 100}%)"
                        )
                    elif pct_missing > 5:
                        warnings.append(f"Column '{col}' has {pct_missing:.1f}% missing data")

        # Check for outliers
        if self.config.check_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                col_data = df[col].dropna()

                if len(col_data) == 0:
                    continue

                # Z-score method
                mean = col_data.mean()
                std = col_data.std()

                if std == 0:
                    warnings.append(f"Column '{col}' has zero variance")
                    continue

                z_scores = np.abs((col_data - mean) / std)
                outlier_mask = z_scores > self.config.outlier_std

                if outlier_mask.any():
                    outlier_indices = col_data[outlier_mask].index.tolist()
                    outliers[col] = outlier_indices

                    n_outliers = len(outlier_indices)
                    pct_outliers = (n_outliers / len(col_data)) * 100

                    if pct_outliers > 10:
                        warnings.append(
                            f"Column '{col}' has {n_outliers} outliers ({pct_outliers:.1f}%)"
                        )

        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate rows")

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has only one unique value")

        # Determine if validation passed
        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            n_subjects=n_subjects,
            n_features=n_features,
            missing_data=missing_data,
            outliers=outliers,
            warnings=warnings,
            errors=errors,
        )

    def check_subject_ids(
        self, imaging_df: pd.DataFrame, metadata_df: pd.DataFrame, id_column: str = "subject_id"
    ) -> List[str]:
        """
        Check consistency of subject IDs between imaging and metadata.

        Args:
            imaging_df: DataFrame with imaging data
            metadata_df: DataFrame with metadata
            id_column: Column name for subject IDs

        Returns:
            List of mismatched subject IDs
        """
        imaging_ids = set(imaging_df.index)
        metadata_ids = set(metadata_df[id_column])

        # Find mismatches
        only_in_imaging = imaging_ids - metadata_ids
        only_in_metadata = metadata_ids - imaging_ids

        mismatches = []

        if only_in_imaging:
            mismatches.append(
                f"{len(only_in_imaging)} subjects in imaging but not in metadata: "
                f"{list(only_in_imaging)[:5]}"
            )

        if only_in_metadata:
            mismatches.append(
                f"{len(only_in_metadata)} subjects in metadata but not in imaging: "
                f"{list(only_in_metadata)[:5]}"
            )

        return mismatches

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Impute missing values in dataset.

        Args:
            df: DataFrame with missing values
            method: Imputation method ('mean', 'median', 'zero', 'drop')
            columns: Optional list of columns to impute (default: all numeric)

        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        if method == "mean":
            for col in columns:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
        elif method == "median":
            for col in columns:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        elif method == "zero":
            df_imputed[columns] = df_imputed[columns].fillna(0)
        elif method == "drop":
            df_imputed = df_imputed.dropna(subset=columns)
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        return df_imputed

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Remove outliers from dataset.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            columns: Columns to check (default: all numeric)

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        if method == "zscore":
            # Z-score method
            for col in columns:
                col_data = df_clean[col].dropna()
                if len(col_data) == 0:
                    continue

                mean = col_data.mean()
                std = col_data.std()

                if std == 0:
                    continue

                z_scores = np.abs((df_clean[col] - mean) / std)
                df_clean = df_clean[z_scores <= threshold]

        elif method == "iqr":
            # IQR method
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]
        else:
            raise ValueError(f"Unknown method: {method}")

        return df_clean
