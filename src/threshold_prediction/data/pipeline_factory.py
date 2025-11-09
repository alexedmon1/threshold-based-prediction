"""
Factory for creating data preparation pipelines.

Supports both human (FreeSurfer-based) and animal (label map-based) pipelines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from threshold_prediction.utils.config import DataPrepConfig


class DataPipeline(ABC):
    """Abstract base class for data preparation pipelines."""

    def __init__(self, config: DataPrepConfig):
        """
        Initialize data pipeline.

        Args:
            config: DataPrepConfig object with pipeline configuration
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data from raw inputs.

        Returns:
            DataFrame with standardized format (subjects × ROIs)
        """
        pass

    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate prepared data.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        pass

    def save_data(self, output_path: Union[str, Path]):
        """
        Save prepared data to CSV.

        Args:
            output_path: Path to save CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Run prepare_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path)

    def run(self, output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Run complete pipeline: prepare, validate, and optionally save.

        Args:
            output_path: Optional path to save output

        Returns:
            Prepared and validated DataFrame
        """
        # Prepare data
        self.data = self.prepare_data()

        # Validate if enabled
        if self.config.validation.check_missing or self.config.validation.check_outliers:
            self.validate_data(self.data)

        # Save if path provided
        if output_path is not None:
            self.save_data(output_path)

        return self.data


class HumanPipeline(DataPipeline):
    """Pipeline for human neuroimaging data using FreeSurfer."""

    def __init__(self, config: DataPrepConfig):
        """
        Initialize human data pipeline.

        Args:
            config: DataPrepConfig with pipeline.type='human'
        """
        super().__init__(config)

        if config.pipeline.type != "human":
            raise ValueError(f"Expected pipeline.type='human', got '{config.pipeline.type}'")

        if config.human is None:
            raise ValueError("FreeSurferConfig required for human pipeline")

        self.fs_config = config.human

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data from FreeSurfer outputs.

        Returns:
            DataFrame with subjects × ROIs
        """
        from threshold_prediction.data.human.freesurfer_parser import FreeSurferParser

        parser = FreeSurferParser(
            subjects_dir=self.fs_config.subjects_dir,
            measures=self.fs_config.measures,
            atlases=self.fs_config.atlases,
        )

        # Parse subjects
        if isinstance(self.fs_config.subjects_list, Path):
            with open(self.fs_config.subjects_list) as f:
                subjects = [line.strip() for line in f if line.strip()]
        elif isinstance(self.fs_config.subjects_list, list):
            subjects = self.fs_config.subjects_list
        else:
            # Auto-detect subjects in directory
            subjects = [
                d.name
                for d in self.fs_config.subjects_dir.iterdir()
                if d.is_dir() and (d / "stats").exists()
            ]

        # Parse all subjects
        df = parser.parse_multiple_subjects(subjects)

        # Merge with metadata if provided
        if self.config.standardization.metadata is not None:
            metadata = pd.read_csv(self.config.standardization.metadata)
            df = pd.merge(
                df,
                metadata,
                left_index=True,
                right_on=self.config.standardization.subject_id_column,
                how="inner",
            )
            df.set_index(self.config.standardization.subject_id_column, inplace=True)

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate human data."""
        from threshold_prediction.data.validation import DataValidator

        validator = DataValidator(self.config.validation)
        report = validator.validate(df)

        if not report.is_valid:
            raise ValueError(f"Data validation failed:\n{report}")

        return True


class AnimalPipeline(DataPipeline):
    """Pipeline for animal neuroimaging data using label maps."""

    def __init__(self, config: DataPrepConfig):
        """
        Initialize animal data pipeline.

        Args:
            config: DataPrepConfig with pipeline.type='animal'
        """
        super().__init__(config)

        if config.pipeline.type != "animal":
            raise ValueError(f"Expected pipeline.type='animal', got '{config.pipeline.type}'")

        if config.animal is None:
            raise ValueError("AnimalConfig required for animal pipeline")

        self.animal_config = config.animal

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data from animal images and label maps.

        Returns:
            DataFrame with subjects × ROIs
        """
        from threshold_prediction.data.animal.labelmap_extractor import LabelMapExtractor
        from threshold_prediction.data.animal.atlas_registry import AtlasRegistry

        # Get atlas
        if self.animal_config.atlas.name is not None:
            # Use pre-configured atlas
            registry = AtlasRegistry()
            atlas_info = registry.get_atlas(
                species=self.config.pipeline.species, atlas_name=self.animal_config.atlas.name
            )
            extractor = LabelMapExtractor(
                atlas_path=atlas_info["path"], labels_mapping=atlas_info["labels"]
            )
        else:
            # Use custom atlas
            if self.animal_config.atlas.path is None or self.animal_config.atlas.labels_csv is None:
                raise ValueError(
                    "Either atlas.name or both atlas.path and atlas.labels_csv must be provided"
                )

            # Load labels mapping
            labels_df = pd.read_csv(self.animal_config.atlas.labels_csv)
            labels_mapping = dict(zip(labels_df["label_id"], labels_df["region_name"]))

            extractor = LabelMapExtractor(
                atlas_path=self.animal_config.atlas.path, labels_mapping=labels_mapping
            )

        # Extract ROI statistics from images
        # Support multiple modalities
        all_data = []

        for modality_name in ["r1", "r2star", "t1"]:
            pattern_key = f"{modality_name}_pattern"
            if hasattr(self.animal_config.images, pattern_key):
                pattern = getattr(self.animal_config.images, pattern_key)
                if pattern is not None:
                    df = extractor.extract_multi_subject(
                        image_pattern=pattern,
                        statistics=self.animal_config.statistics,
                        modality_prefix=modality_name,
                    )
                    all_data.append(df)

        # Combine modalities
        if len(all_data) == 0:
            # Try generic pattern
            if self.animal_config.images.pattern is not None:
                df = extractor.extract_multi_subject(
                    image_pattern=self.animal_config.images.pattern,
                    statistics=self.animal_config.statistics,
                )
                all_data.append(df)

        if len(all_data) == 0:
            raise ValueError("No image patterns provided in configuration")

        # Merge all modalities
        combined_df = all_data[0]
        for df in all_data[1:]:
            combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how="outer")

        # Merge with metadata if provided
        if self.config.standardization.metadata is not None:
            metadata = pd.read_csv(self.config.standardization.metadata)
            combined_df = pd.merge(
                combined_df,
                metadata,
                left_index=True,
                right_on=self.config.standardization.subject_id_column,
                how="inner",
            )
            combined_df.set_index(self.config.standardization.subject_id_column, inplace=True)

        return combined_df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate animal data."""
        from threshold_prediction.data.validation import DataValidator

        validator = DataValidator(self.config.validation)
        report = validator.validate(df)

        if not report.is_valid:
            raise ValueError(f"Data validation failed:\n{report}")

        return True


class DataPipelineFactory:
    """Factory for creating appropriate data preparation pipeline."""

    @staticmethod
    def create(config: DataPrepConfig) -> DataPipeline:
        """
        Create pipeline based on configuration.

        Args:
            config: DataPrepConfig object

        Returns:
            HumanPipeline or AnimalPipeline instance

        Raises:
            ValueError: If pipeline type is unknown
        """
        if config.pipeline.type == "human":
            return HumanPipeline(config)
        elif config.pipeline.type == "animal":
            return AnimalPipeline(config)
        else:
            raise ValueError(
                f"Unknown pipeline type: {config.pipeline.type}. "
                f"Must be 'human' or 'animal'"
            )

    @staticmethod
    def from_config_file(config_path: Union[str, Path]) -> DataPipeline:
        """
        Create pipeline from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured pipeline instance
        """
        from threshold_prediction.utils.config import ConfigManager

        config = ConfigManager.load_data_prep_config(config_path)
        return DataPipelineFactory.create(config)
