"""
Configuration management for threshold prediction package.

Uses Pydantic for validation and type checking of configuration files.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class AtlasConfig(BaseModel):
    """Configuration for brain atlas."""

    type: Literal["labelmap", "freesurfer"] = Field(
        description="Type of atlas (labelmap for animals, freesurfer for humans)"
    )
    name: Optional[str] = Field(
        default=None, description="Pre-configured atlas name (e.g., 'sigma', 'waxholm')"
    )
    path: Optional[Path] = Field(default=None, description="Path to custom atlas file")
    labels_csv: Optional[Path] = Field(
        default=None, description="Path to CSV mapping label IDs to region names"
    )

    @field_validator("path", "labels_csv", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v


class ImageConfig(BaseModel):
    """Configuration for image data."""

    r1_pattern: Optional[str] = Field(
        default=None, description="Glob pattern for R1 images (e.g., 'data/*/R1.nii.gz')"
    )
    r2star_pattern: Optional[str] = Field(
        default=None, description="Glob pattern for R2* images"
    )
    t1_pattern: Optional[str] = Field(default=None, description="Glob pattern for T1 images")
    pattern: Optional[str] = Field(
        default=None, description="Generic pattern if using single modality"
    )
    modalities: Optional[List[Dict[str, str]]] = Field(
        default=None, description="List of modality configurations"
    )


class AnimalConfig(BaseModel):
    """Configuration for animal data pipeline."""

    atlas: AtlasConfig
    images: ImageConfig
    statistics: List[str] = Field(
        default=["mean", "median", "std", "variance", "skew"],
        description="Statistical measures to extract",
    )
    preprocessing: Dict = Field(
        default_factory=dict, description="Preprocessing options"
    )


class FreeSurferConfig(BaseModel):
    """Configuration for FreeSurfer/human data pipeline."""

    subjects_dir: Path
    subjects_list: Optional[Union[Path, List[str]]] = Field(
        default=None, description="Path to subjects list file or list of subject IDs"
    )
    measures: List[str] = Field(
        default=["volume", "thickness", "area"], description="FreeSurfer measures to extract"
    )
    atlases: List[str] = Field(
        default=["aseg", "aparc"], description="FreeSurfer atlases to use"
    )

    @field_validator("subjects_dir", mode="before")
    @classmethod
    def convert_subjects_dir(cls, v):
        if not isinstance(v, Path):
            return Path(v)
        return v


class StandardizationConfig(BaseModel):
    """Configuration for data standardization."""

    imaging_data: Optional[Path] = None
    metadata: Optional[Path] = None
    subject_id_column: str = Field(default="subject_id", description="Column name for subject IDs")
    target_variables: List[str] = Field(
        default_factory=list, description="Target variables for prediction"
    )
    output: Path = Field(default=Path("analysis_ready.csv"))
    validate: bool = Field(default=True, description="Run data validation")

    @field_validator("imaging_data", "metadata", "output", mode="before")
    @classmethod
    def convert_paths(cls, v):
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v


class ValidationConfig(BaseModel):
    """Configuration for data validation."""

    check_missing: bool = True
    missing_threshold: float = Field(
        default=0.1, description="Maximum allowed proportion of missing data"
    )
    check_outliers: bool = True
    outlier_std: float = Field(default=3.0, description="Standard deviations for outlier detection")


class ModelConfig(BaseModel):
    """Configuration for SVM model."""

    pca_variance: float = Field(default=0.90, ge=0.0, le=1.0, description="PCA variance to retain")
    cv_method: Literal["kfold", "loo"] = Field(default="kfold", description="Cross-validation method")
    cv_folds: int = Field(default=10, ge=2, description="Number of CV folds")
    svm_kernel: Literal["linear", "rbf", "poly"] = Field(default="linear")
    class_weight: Literal["balanced", "none"] = Field(default="balanced")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class EnsembleConfig(BaseModel):
    """Configuration for ensemble methods."""

    statistics: List[str] = Field(
        default=["median", "variance", "skew"], description="Statistics for bagging"
    )
    voting: Literal["majority", "weighted"] = Field(default="majority")


class RegionConfig(BaseModel):
    """Configuration for brain region selection."""

    use_regions: Union[str, List[str]] = Field(
        default="all", description="Regions to use ('all' or list of region names)"
    )
    exclude_regions: List[str] = Field(default_factory=list, description="Regions to exclude")


class AnalysisConfig(BaseModel):
    """Configuration for threshold analysis."""

    target_variable: str
    threshold_range: List[float] = Field(description="[min, max] for threshold scanning")
    threshold_step: float = Field(description="Step size for threshold scanning")

    @field_validator("threshold_range")
    @classmethod
    def validate_threshold_range(cls, v):
        if len(v) != 2:
            raise ValueError("threshold_range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("threshold_range min must be less than max")
        return v


class OutputConfig(BaseModel):
    """Configuration for outputs."""

    results_dir: Path = Field(default=Path("./results/"))
    save_models: bool = True
    generate_plots: bool = True
    export_csv: bool = True
    export_json: bool = True

    @field_validator("results_dir", mode="before")
    @classmethod
    def convert_results_dir(cls, v):
        if not isinstance(v, Path):
            return Path(v)
        return v


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    type: Literal["animal", "human"]
    species: Optional[str] = Field(default=None, description="Species for animal pipeline")


class DataPrepConfig(BaseModel):
    """Complete configuration for data preparation."""

    pipeline: PipelineConfig
    animal: Optional[AnimalConfig] = None
    human: Optional[FreeSurferConfig] = None
    standardization: StandardizationConfig = Field(default_factory=StandardizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @field_validator("animal", "human", mode="after")
    @classmethod
    def validate_pipeline_config(cls, v, info):
        pipeline_type = info.data.get("pipeline").type if "pipeline" in info.data else None
        field_name = info.field_name

        if pipeline_type == "animal" and field_name == "animal" and v is None:
            raise ValueError("animal config required when pipeline.type is 'animal'")
        if pipeline_type == "human" and field_name == "human" and v is None:
            raise ValueError("human config required when pipeline.type is 'human'")
        return v


class ThresholdConfig(BaseModel):
    """Complete configuration for threshold analysis."""

    data: Dict[str, Union[str, Path]]
    analysis: AnalysisConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    regions: RegionConfig = Field(default_factory=RegionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class ConfigManager:
    """Manages loading and validation of configuration files."""

    @staticmethod
    def load_yaml(config_path: Union[str, Path]) -> dict:
        """Load YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return config_dict

    @staticmethod
    def load_data_prep_config(config_path: Union[str, Path]) -> DataPrepConfig:
        """Load and validate data preparation configuration."""
        config_dict = ConfigManager.load_yaml(config_path)
        return DataPrepConfig(**config_dict)

    @staticmethod
    def load_threshold_config(config_path: Union[str, Path]) -> ThresholdConfig:
        """Load and validate threshold analysis configuration."""
        config_dict = ConfigManager.load_yaml(config_path)
        return ThresholdConfig(**config_dict)

    @staticmethod
    def save_config(config: Union[DataPrepConfig, ThresholdConfig], output_path: Union[str, Path]):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        config_dict = config.model_dump(mode="python")

        # Convert Path objects to strings for YAML
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        config_dict = convert_paths(config_dict)

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
