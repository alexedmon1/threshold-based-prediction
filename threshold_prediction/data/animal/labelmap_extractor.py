"""
Extract ROI statistics from neuroimaging data using label maps.

Supports extraction from animal neuroimaging data (rats, mice, NHP) using
atlas-based parcellation.
"""

from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm


class LabelMapExtractor:
    """Extract ROI statistics from images using label map atlases."""

    def __init__(self, atlas_path: Path, labels_mapping: Dict[int, str]):
        """
        Initialize label map extractor.

        Args:
            atlas_path: Path to NIfTI label map atlas
            labels_mapping: Dictionary mapping label IDs to region names
        """
        self.atlas_path = Path(atlas_path)
        self.labels_mapping = labels_mapping

        # Load atlas
        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Atlas not found: {self.atlas_path}")

        self.atlas_img = nib.load(str(self.atlas_path))
        self.atlas_data = self.atlas_img.get_fdata()

        # Get unique labels in atlas
        self.unique_labels = np.unique(self.atlas_data[self.atlas_data > 0]).astype(int)

        # Validate labels
        missing_labels = set(self.unique_labels) - set(self.labels_mapping.keys())
        if missing_labels:
            print(f"Warning: {len(missing_labels)} labels in atlas not found in mapping")

    def extract_roi_statistics(
        self,
        image_path: Path,
        statistics: List[str] = ["mean", "median", "std"],
    ) -> pd.DataFrame:
        """
        Extract ROI statistics from a single image.

        Args:
            image_path: Path to NIfTI image
            statistics: List of statistics to compute. Options:
                - 'mean': Mean value
                - 'median': Median value
                - 'std': Standard deviation
                - 'variance': Variance
                - 'skew': Skewness
                - 'kurtosis': Kurtosis
                - 'min': Minimum value
                - 'max': Maximum value
                - 'percentile_10': 10th percentile
                - 'percentile_90': 90th percentile
                - 'range': Range (max - min)

        Returns:
            DataFrame with one row per ROI and columns for each statistic
        """
        # Load image
        img = nib.load(str(image_path))
        img_data = img.get_fdata()

        # Check dimensions match
        if img_data.shape != self.atlas_data.shape:
            raise ValueError(
                f"Image shape {img_data.shape} does not match "
                f"atlas shape {self.atlas_data.shape}"
            )

        # Extract statistics for each ROI
        results = []

        for label_id in self.unique_labels:
            # Get voxels for this ROI
            mask = self.atlas_data == label_id
            roi_values = img_data[mask]

            # Skip if no voxels
            if len(roi_values) == 0:
                continue

            # Compute statistics
            roi_stats = {"label_id": int(label_id), "region_name": self.labels_mapping.get(label_id, f"label_{label_id}")}

            for stat in statistics:
                if stat == "mean":
                    roi_stats["mean"] = np.mean(roi_values)
                elif stat == "median":
                    roi_stats["median"] = np.median(roi_values)
                elif stat == "std":
                    roi_stats["std"] = np.std(roi_values)
                elif stat == "variance":
                    roi_stats["variance"] = np.var(roi_values)
                elif stat == "skew":
                    roi_stats["skew"] = scipy_stats.skew(roi_values)
                elif stat == "kurtosis":
                    roi_stats["kurtosis"] = scipy_stats.kurtosis(roi_values)
                elif stat == "min":
                    roi_stats["min"] = np.min(roi_values)
                elif stat == "max":
                    roi_stats["max"] = np.max(roi_values)
                elif stat == "percentile_10":
                    roi_stats["percentile_10"] = np.percentile(roi_values, 10)
                elif stat == "percentile_90":
                    roi_stats["percentile_90"] = np.percentile(roi_values, 90)
                elif stat == "range":
                    roi_stats["range"] = np.max(roi_values) - np.min(roi_values)
                elif stat == "count":
                    roi_stats["count"] = len(roi_values)
                else:
                    print(f"Warning: Unknown statistic '{stat}', skipping")

            results.append(roi_stats)

        return pd.DataFrame(results)

    def extract_multi_subject(
        self,
        image_pattern: str,
        statistics: List[str] = ["mean", "median", "std"],
        modality_prefix: Optional[str] = None,
        subject_id_from_path: bool = True,
    ) -> pd.DataFrame:
        """
        Extract ROI statistics from multiple subjects.

        Args:
            image_pattern: Glob pattern for images (e.g., "data/subjects/*/R1.nii.gz")
            statistics: List of statistics to compute
            modality_prefix: Optional prefix for column names (e.g., "r1_")
            subject_id_from_path: Extract subject ID from file path

        Returns:
            DataFrame with subjects as rows and ROI statistics as columns
        """
        # Find all matching images
        image_files = sorted(glob(image_pattern))

        if len(image_files) == 0:
            raise ValueError(f"No files found matching pattern: {image_pattern}")

        print(f"Processing {len(image_files)} images...")

        # Process each subject
        all_subjects = []

        for image_path in tqdm(image_files, desc="Extracting ROIs"):
            # Get subject ID
            if subject_id_from_path:
                # Extract subject ID from path (assumes format: .../subject_id/...)
                subject_id = Path(image_path).parent.name
            else:
                subject_id = Path(image_path).stem

            # Extract statistics
            df_stats = self.extract_roi_statistics(Path(image_path), statistics)

            # Pivot to wide format: one row per subject
            # Columns: region_stat (e.g., striatum_mean, striatum_median)
            subject_data = {"subject_id": subject_id}

            for _, row in df_stats.iterrows():
                region_name = row["region_name"]
                for stat in statistics:
                    if stat in row:
                        col_name = f"{region_name}_{stat}"
                        if modality_prefix:
                            col_name = f"{modality_prefix}_{col_name}"
                        subject_data[col_name] = row[stat]

            all_subjects.append(subject_data)

        # Combine all subjects
        df_combined = pd.DataFrame(all_subjects)
        df_combined.set_index("subject_id", inplace=True)

        return df_combined

    def extract_subset_regions(
        self,
        image_path: Path,
        region_names: List[str],
        statistics: List[str] = ["mean"],
    ) -> pd.DataFrame:
        """
        Extract statistics for a subset of regions.

        Args:
            image_path: Path to NIfTI image
            region_names: List of region names to extract
            statistics: List of statistics to compute

        Returns:
            DataFrame with statistics for specified regions only
        """
        # Get label IDs for requested regions
        label_ids = []
        for region_name in region_names:
            # Find matching label ID
            matching_ids = [
                label_id
                for label_id, name in self.labels_mapping.items()
                if name == region_name
            ]
            if len(matching_ids) > 0:
                label_ids.extend(matching_ids)
            else:
                print(f"Warning: Region '{region_name}' not found in atlas")

        if len(label_ids) == 0:
            raise ValueError("No matching regions found in atlas")

        # Extract all statistics
        df_all = self.extract_roi_statistics(image_path, statistics)

        # Filter to requested regions
        df_subset = df_all[df_all["label_id"].isin(label_ids)]

        return df_subset

    def get_atlas_info(self) -> Dict:
        """
        Get information about the loaded atlas.

        Returns:
            Dictionary with atlas information
        """
        return {
            "path": str(self.atlas_path),
            "shape": self.atlas_data.shape,
            "num_labels": len(self.unique_labels),
            "labels": self.unique_labels.tolist(),
            "regions": list(self.labels_mapping.values()),
        }


class MultiModalityExtractor:
    """Extract statistics from multiple imaging modalities."""

    def __init__(self, atlas_path: Path, labels_mapping: Dict[int, str]):
        """
        Initialize multi-modality extractor.

        Args:
            atlas_path: Path to label map atlas
            labels_mapping: Label ID to region name mapping
        """
        self.extractor = LabelMapExtractor(atlas_path, labels_mapping)

    def extract_multiple_modalities(
        self,
        modality_patterns: Dict[str, str],
        statistics: List[str] = ["mean", "median"],
    ) -> pd.DataFrame:
        """
        Extract statistics from multiple modalities.

        Args:
            modality_patterns: Dictionary mapping modality names to glob patterns
                Example: {"r1": "data/*/R1.nii.gz", "r2star": "data/*/R2star.nii.gz"}
            statistics: Statistics to compute

        Returns:
            Combined DataFrame with all modalities
        """
        dfs = []

        for modality_name, pattern in modality_patterns.items():
            print(f"\nProcessing {modality_name}...")
            df = self.extractor.extract_multi_subject(
                image_pattern=pattern,
                statistics=statistics,
                modality_prefix=modality_name,
            )
            dfs.append(df)

        # Merge all modalities
        df_combined = dfs[0]
        for df in dfs[1:]:
            df_combined = pd.merge(
                df_combined,
                df,
                left_index=True,
                right_index=True,
                how="outer",
            )

        return df_combined
