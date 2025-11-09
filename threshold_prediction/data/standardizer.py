"""
Data standardization and formatting utilities.

Supports:
- Direct CSV input for pre-formatted data
- Merging imaging data with metadata
- Format validation and conversion
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from threshold_prediction.data.validation import DataValidator, ValidationReport
from threshold_prediction.utils.config import ValidationConfig


class DataStandardizer:
    """Standardize data into analysis-ready format."""

    @staticmethod
    def load_from_csv(
        csv_path: Union[str, Path],
        subject_id_column: str = "subject_id",
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load data directly from CSV file.

        This is useful for:
        - Pre-formatted data from other sources
        - Manual ROI measurements
        - Non-imaging data
        - Studies with partial brain coverage

        Expected CSV format:
        - First column: Subject IDs
        - Remaining columns: ROI measurements and/or metadata
        - Column names should be descriptive (e.g., 'striatum_volume', 'age', 'dose')

        Args:
            csv_path: Path to CSV file
            subject_id_column: Name of the subject ID column
            validate: Run data validation

        Returns:
            DataFrame with standardized format

        Example CSV:
        ```
        subject_id,striatum_left,striatum_right,age,dose_mg_kg
        rat_001,12.5,12.8,90,0.0
        rat_002,11.2,11.5,95,5.0
        rat_003,10.8,11.0,92,10.0
        ```
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)

        # Set subject ID as index if column exists
        if subject_id_column in df.columns:
            df.set_index(subject_id_column, inplace=True)
        else:
            # Try to find subject ID column (case-insensitive)
            possible_names = [
                "subject",
                "subjectid",
                "subject_id",
                "id",
                "animal_id",
                "animalid",
                "code",
            ]
            found = False
            for col in df.columns:
                if col.lower() in possible_names:
                    df.set_index(col, inplace=True)
                    found = True
                    print(f"Using column '{col}' as subject ID")
                    break

            if not found:
                print(
                    "Warning: No subject ID column found. Using first column or row numbers."
                )
                if len(df.columns) > 0:
                    df.set_index(df.columns[0], inplace=True)

        # Validate if requested
        if validate:
            validator = DataValidator(ValidationConfig())
            report = validator.validate(df)

            if not report.is_valid:
                print("Warning: Data validation found issues:")
                print(report)

        return df

    @staticmethod
    def merge_with_metadata(
        imaging_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        subject_id_column: str = "subject_id",
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Merge imaging data with metadata.

        Args:
            imaging_df: DataFrame with imaging/ROI measurements (subjects as index)
            metadata_df: DataFrame with metadata (subject IDs in column)
            subject_id_column: Column name in metadata for subject IDs
            how: Merge type ('inner', 'left', 'outer')

        Returns:
            Merged DataFrame
        """
        # Ensure metadata has subject ID column
        if subject_id_column not in metadata_df.columns:
            raise ValueError(f"Metadata must have '{subject_id_column}' column")

        # Merge
        merged = pd.merge(
            imaging_df,
            metadata_df,
            left_index=True,
            right_on=subject_id_column,
            how=how,
        )

        # Set index to subject ID
        if subject_id_column != merged.index.name:
            merged.set_index(subject_id_column, inplace=True)

        return merged

    @staticmethod
    def create_data_template(
        output_path: Union[str, Path],
        n_rois: int = 5,
        n_subjects: int = 3,
        include_metadata: bool = True,
    ):
        """
        Create a template CSV file showing the expected data format.

        Args:
            output_path: Path to save template CSV
            n_rois: Number of example ROIs to include
            n_subjects: Number of example subjects to include
            include_metadata: Include metadata columns (age, dose, etc.)
        """
        output_path = Path(output_path)

        # Create example data
        data = {"subject_id": [f"subject_{i+1:03d}" for i in range(n_subjects)]}

        # Add ROI columns
        roi_names = [f"roi_{i+1}" for i in range(n_rois)]
        for roi in roi_names:
            data[roi] = [10.0 + i * 0.5 for i in range(n_subjects)]

        # Add metadata columns
        if include_metadata:
            data["age_days"] = [80, 85, 90]
            data["dose_mg_kg"] = [0.0, 5.0, 10.0]
            data["treatment_group"] = ["control", "low", "high"]

        df = pd.DataFrame(data)

        # Write with comments
        with open(output_path, "w") as f:
            f.write("# Data template for threshold prediction analysis\n")
            f.write("# \n")
            f.write("# Required columns:\n")
            f.write("#   - subject_id: Unique identifier for each subject\n")
            f.write("#   - ROI measurements: One column per brain region/measurement\n")
            f.write("# \n")
            f.write("# Optional columns:\n")
            f.write("#   - Metadata: age, dose, treatment group, etc.\n")
            f.write("#   - Target variables for threshold analysis\n")
            f.write("# \n")
            f.write("# Notes:\n")
            f.write("#   - Use descriptive column names (e.g., 'striatum_volume_mm3')\n")
            f.write("#   - Missing values will be imputed (mean by default)\n")
            f.write("#   - All numeric ROI columns will be used for analysis\n")
            f.write("# \n\n")

        # Append dataframe
        df.to_csv(output_path, mode="a", index=False)

        print(f"Template created: {output_path}")

    @staticmethod
    def validate_format(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_subjects: int = 5,
        min_features: int = 2,
    ) -> ValidationReport:
        """
        Validate data format for analysis.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_subjects: Minimum number of subjects required
            min_features: Minimum number of features required

        Returns:
            ValidationReport with validation results
        """
        warnings = []
        errors = []

        # Check dimensions
        n_subjects = len(df)
        n_features = len(df.columns)

        if n_subjects < min_subjects:
            errors.append(
                f"Insufficient subjects: {n_subjects} (minimum: {min_subjects})"
            )

        if n_features < min_features:
            errors.append(
                f"Insufficient features: {n_features} (minimum: {min_features})"
            )

        # Check required columns
        if required_columns is not None:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            errors.append("No numeric columns found for analysis")
        elif len(numeric_cols) < min_features:
            warnings.append(
                f"Only {len(numeric_cols)} numeric columns found "
                f"(recommended: >={min_features})"
            )

        # Standard validation
        validator = DataValidator(ValidationConfig())
        report = validator.validate(df)

        # Combine results
        report.warnings.extend(warnings)
        report.errors.extend(errors)
        report.is_valid = report.is_valid and len(errors) == 0

        return report

    @staticmethod
    def export_for_analysis(
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        statistics: Optional[List[str]] = None,
        prefix: str = "data",
    ):
        """
        Export data in formats ready for threshold analysis.

        If statistics are specified, creates separate files for each statistic
        (matching legacy code pattern with separate CSV files for median, variance, skew, etc.)

        Args:
            df: Input DataFrame
            output_dir: Directory to save output files
            statistics: Optional list of statistics to split by
            prefix: Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if statistics is None:
            # Save single file
            output_path = output_dir / f"{prefix}.csv"
            df.to_csv(output_path)
            print(f"Saved: {output_path}")
        else:
            # Save separate file for each statistic
            for stat in statistics:
                # Find columns containing this statistic
                stat_cols = [col for col in df.columns if f"_{stat}" in col.lower()]

                if len(stat_cols) > 0:
                    df_stat = df[stat_cols].copy()

                    # Remove statistic suffix from column names
                    df_stat.columns = [
                        col.replace(f"_{stat}", "").replace(f"_{stat.upper()}", "")
                        for col in df_stat.columns
                    ]

                    output_path = output_dir / f"{prefix}_{stat}.csv"
                    df_stat.to_csv(output_path)
                    print(f"Saved: {output_path}")


class CSVPipeline:
    """Pipeline for handling pre-formatted CSV data."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        subject_id_column: str = "subject_id",
    ):
        """
        Initialize CSV pipeline.

        Args:
            csv_path: Path to CSV with ROI data
            metadata_path: Optional path to metadata CSV
            subject_id_column: Column name for subject IDs
        """
        self.csv_path = Path(csv_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.subject_id_column = subject_id_column
        self.data: Optional[pd.DataFrame] = None

    def prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data from CSV.

        Returns:
            Prepared DataFrame
        """
        # Load main data
        self.data = DataStandardizer.load_from_csv(
            self.csv_path,
            subject_id_column=self.subject_id_column,
            validate=True,
        )

        # Merge with metadata if provided
        if self.metadata_path is not None and self.metadata_path.exists():
            metadata = pd.read_csv(self.metadata_path)
            self.data = DataStandardizer.merge_with_metadata(
                self.data,
                metadata,
                subject_id_column=self.subject_id_column,
            )

        return self.data

    def validate(self) -> ValidationReport:
        """Validate prepared data."""
        if self.data is None:
            raise ValueError("No data loaded. Run prepare_data() first.")

        return DataStandardizer.validate_format(self.data)

    def save(self, output_path: Union[str, Path]):
        """Save prepared data."""
        if self.data is None:
            raise ValueError("No data loaded. Run prepare_data() first.")

        self.data.to_csv(output_path)
