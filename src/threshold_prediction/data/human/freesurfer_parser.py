"""
Parse FreeSurfer outputs to extract neuroimaging statistics.

Supports parsing of aseg.stats, aparc.stats, and other FreeSurfer output files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


class FreeSurferParser:
    """Parser for FreeSurfer statistics files."""

    def __init__(
        self,
        subjects_dir: Path,
        measures: List[str] = ["volume", "thickness", "area"],
        atlases: List[str] = ["aseg", "aparc"],
    ):
        """
        Initialize FreeSurfer parser.

        Args:
            subjects_dir: Path to FreeSurfer SUBJECTS_DIR
            measures: List of measures to extract ('volume', 'thickness', 'area', 'mean', 'std')
            atlases: List of atlases to parse ('aseg', 'aparc', 'aparc.a2009s', etc.)
        """
        self.subjects_dir = Path(subjects_dir)
        self.measures = measures
        self.atlases = atlases

        if not self.subjects_dir.exists():
            raise FileNotFoundError(f"Subjects directory not found: {self.subjects_dir}")

    def parse_aseg_stats(self, subject_id: str) -> pd.DataFrame:
        """
        Parse aseg.stats file for subcortical volumes.

        Args:
            subject_id: Subject ID

        Returns:
            DataFrame with structure names and volumes
        """
        stats_file = self.subjects_dir / subject_id / "stats" / "aseg.stats"

        if not stats_file.exists():
            raise FileNotFoundError(f"aseg.stats not found for {subject_id}: {stats_file}")

        # Parse stats file
        data = []

        with open(stats_file, "r") as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith("#") or not line.strip():
                    continue

                # Parse data lines
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        index = int(parts[0])
                        seg_id = int(parts[1])
                        n_voxels = int(parts[2])
                        volume_mm3 = float(parts[3])
                        structure_name = parts[4]

                        # Optional: mean intensity and std
                        mean_intensity = float(parts[5]) if len(parts) > 5 else None
                        std_intensity = float(parts[6]) if len(parts) > 6 else None

                        data.append(
                            {
                                "structure": structure_name,
                                "volume_mm3": volume_mm3,
                                "n_voxels": n_voxels,
                                "mean": mean_intensity,
                                "std": std_intensity,
                            }
                        )
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue

        return pd.DataFrame(data)

    def parse_aparc_stats(self, subject_id: str, hemi: str = "lh") -> pd.DataFrame:
        """
        Parse aparc.stats file for cortical parcellation.

        Args:
            subject_id: Subject ID
            hemi: Hemisphere ('lh' or 'rh')

        Returns:
            DataFrame with ROI names and measurements
        """
        stats_file = self.subjects_dir / subject_id / "stats" / f"{hemi}.aparc.stats"

        if not stats_file.exists():
            raise FileNotFoundError(
                f"{hemi}.aparc.stats not found for {subject_id}: {stats_file}"
            )

        # Parse stats file
        data = []

        with open(stats_file, "r") as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith("#") or not line.strip():
                    continue

                # Parse data lines
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        structure_name = parts[0]
                        num_vertices = int(parts[1])
                        surface_area = float(parts[2])
                        gray_vol = float(parts[3])
                        thickness_avg = float(parts[4])

                        # Optional fields
                        thickness_std = float(parts[5]) if len(parts) > 5 else None
                        mean_curv = float(parts[6]) if len(parts) > 6 else None
                        gauss_curv = float(parts[7]) if len(parts) > 7 else None
                        fold_index = float(parts[8]) if len(parts) > 8 else None
                        curv_index = float(parts[9]) if len(parts) > 9 else None

                        # Add hemisphere prefix to structure name
                        full_name = f"{hemi}-{structure_name}"

                        data.append(
                            {
                                "structure": full_name,
                                "num_vertices": num_vertices,
                                "surface_area_mm2": surface_area,
                                "gray_vol_mm3": gray_vol,
                                "thickness_avg_mm": thickness_avg,
                                "thickness_std_mm": thickness_std,
                                "mean_curv": mean_curv,
                            }
                        )
                    except (ValueError, IndexError):
                        continue

        return pd.DataFrame(data)

    def parse_subject(self, subject_id: str) -> pd.DataFrame:
        """
        Parse all stats files for a subject.

        Args:
            subject_id: Subject ID

        Returns:
            DataFrame with all measurements in wide format
        """
        all_data = {}

        # Parse aseg if requested
        if "aseg" in self.atlases:
            try:
                df_aseg = self.parse_aseg_stats(subject_id)

                # Add to combined data
                for _, row in df_aseg.iterrows():
                    structure = row["structure"]

                    if "volume" in self.measures and pd.notna(row.get("volume_mm3")):
                        all_data[structure] = row["volume_mm3"]

                    if "mean" in self.measures and pd.notna(row.get("mean")):
                        all_data[f"{structure}_mean"] = row["mean"]

                    if "std" in self.measures and pd.notna(row.get("std")):
                        all_data[f"{structure}_std"] = row["std"]

            except FileNotFoundError as e:
                print(f"Warning: {e}")

        # Parse aparc if requested
        if "aparc" in self.atlases:
            for hemi in ["lh", "rh"]:
                try:
                    df_aparc = self.parse_aparc_stats(subject_id, hemi)

                    for _, row in df_aparc.iterrows():
                        structure = row["structure"]

                        if "volume" in self.measures and pd.notna(row.get("gray_vol_mm3")):
                            all_data[structure] = row["gray_vol_mm3"]

                        if "area" in self.measures and pd.notna(row.get("surface_area_mm2")):
                            all_data[f"{structure}_area"] = row["surface_area_mm2"]

                        if "thickness" in self.measures and pd.notna(row.get("thickness_avg_mm")):
                            all_data[f"{structure}_thickness"] = row["thickness_avg_mm"]

                except FileNotFoundError as e:
                    print(f"Warning: {e}")

        # Convert to DataFrame (single row)
        df = pd.DataFrame([all_data])
        df.index = [subject_id]

        return df

    def parse_multiple_subjects(self, subject_list: List[str]) -> pd.DataFrame:
        """
        Parse stats files for multiple subjects.

        Args:
            subject_list: List of subject IDs

        Returns:
            DataFrame with subjects as rows and measurements as columns
        """
        all_subjects = []

        for subject_id in tqdm(subject_list, desc="Parsing FreeSurfer data"):
            try:
                df = self.parse_subject(subject_id)
                all_subjects.append(df)
            except Exception as e:
                print(f"Error parsing subject {subject_id}: {e}")

        # Combine all subjects
        if len(all_subjects) == 0:
            raise ValueError("No subjects successfully parsed")

        df_combined = pd.concat(all_subjects, axis=0)

        return df_combined

    def get_subject_list(self, exclude_fsaverage: bool = True) -> List[str]:
        """
        Get list of subjects in SUBJECTS_DIR.

        Args:
            exclude_fsaverage: Exclude fsaverage template subjects

        Returns:
            List of subject IDs
        """
        subjects = []

        for subj_dir in self.subjects_dir.iterdir():
            if not subj_dir.is_dir():
                continue

            # Check if valid FreeSurfer subject (has stats directory)
            if not (subj_dir / "stats").exists():
                continue

            # Optionally exclude fsaverage
            if exclude_fsaverage and "fsaverage" in subj_dir.name.lower():
                continue

            subjects.append(subj_dir.name)

        return sorted(subjects)


class FreeSurferStatsExtractor:
    """Extract specific statistics from FreeSurfer data."""

    @staticmethod
    def extract_volumes(subjects_dir: Path, subject_list: List[str]) -> pd.DataFrame:
        """
        Extract only volumes from FreeSurfer data.

        Args:
            subjects_dir: FreeSurfer SUBJECTS_DIR
            subject_list: List of subject IDs

        Returns:
            DataFrame with subject volumes
        """
        parser = FreeSurferParser(
            subjects_dir=subjects_dir, measures=["volume"], atlases=["aseg", "aparc"]
        )

        return parser.parse_multiple_subjects(subject_list)

    @staticmethod
    def extract_thickness(subjects_dir: Path, subject_list: List[str]) -> pd.DataFrame:
        """
        Extract cortical thickness from FreeSurfer data.

        Args:
            subjects_dir: FreeSurfer SUBJECTS_DIR
            subject_list: List of subject IDs

        Returns:
            DataFrame with cortical thickness
        """
        parser = FreeSurferParser(
            subjects_dir=subjects_dir, measures=["thickness"], atlases=["aparc"]
        )

        return parser.parse_multiple_subjects(subject_list)

    @staticmethod
    def extract_all_measures(subjects_dir: Path, subject_list: List[str]) -> pd.DataFrame:
        """
        Extract all available measures from FreeSurfer data.

        Args:
            subjects_dir: FreeSurfer SUBJECTS_DIR
            subject_list: List of subject IDs

        Returns:
            DataFrame with all measurements
        """
        parser = FreeSurferParser(
            subjects_dir=subjects_dir,
            measures=["volume", "thickness", "area", "mean", "std"],
            atlases=["aseg", "aparc"],
        )

        return parser.parse_multiple_subjects(subject_list)
