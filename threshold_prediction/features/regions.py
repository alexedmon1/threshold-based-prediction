"""
Brain region definitions and utilities.

Provides structural and functional groupings of brain regions for:
- FreeSurfer parcellations (human)
- Custom region definitions
- Region filtering and selection
"""

import re
from typing import Dict, List, Optional, Set, Union

import pandas as pd


class BrainRegions:
    """Brain region definitions and utilities."""

    # FreeSurfer structural groupings (from legacy code)
    FREESURFER_STRUCTURAL = {
        "Cortex": "ctx",  # Pattern-based: all regions starting with "ctx"
        "Cerebellum": [
            "Left-Cerebellum-White-Matter",
            "Left-Cerebellum-Cortex",
            "Right-Cerebellum-White-Matter",
            "Right-Cerebellum-Cortex",
        ],
        "BasalGanglia": [
            "Left-Thalamus-Proper",
            "Left-Caudate",
            "Left-Putamen",
            "Left-Pallidum",
            "Right-Thalamus-Proper",
            "Right-Caudate",
            "Right-Putamen",
            "Right-Pallidum",
        ],
        "Ventricles": [
            "Left-Lateral-Ventricle",
            "Left-Inf-Lat-Vent",
            "3rd-Ventricle",
            "4th-Ventricle",
            "CSF",
            "Right-Lateral-Ventricle",
            "Right-Inf-Lat-Vent",
        ],
        "WhiteMatter": [
            "Left-Cerebral-White-Matter",
            "Right-Cerebral-White-Matter",
            "CC_Posterior",
            "CC_Mid_Posterior",
            "CC_Central",
            "CC_Mid_Anterior",
            "CC_Anterior",
        ],
        "InnerBrain": [
            "Brain-Stem",
            "Left-choroid-plexus",
            "Right-choroid-plexus",
            "Left-Hippocampus",
            "Left-Amygdala",
            "Left-VentralDC",
            "Left-Accumbens-area",
            "Right-Hippocampus",
            "Right-Amygdala",
            "Right-VentralDC",
            "Right-Accumbens-area",
        ],
        "Blood": ["Left-vessel", "Right-vessel"],
    }

    # FreeSurfer functional groupings
    FREESURFER_FUNCTIONAL = {
        "Motor": [
            "ctx-lh-precentral",
            "ctx-rh-precentral",
            "ctx-lh-postcentral",
            "ctx-rh-postcentral",
            "Left-Putamen",
            "Right-Putamen",
        ],
        "Limbic": [
            "ctx-lh-cingulate",
            "ctx-rh-cingulate",
            "Left-Hippocampus",
            "Right-Hippocampus",
            "Left-Amygdala",
            "Right-Amygdala",
        ],
        "Executive": [
            "ctx-lh-superiorfrontal",
            "ctx-rh-superiorfrontal",
            "ctx-lh-rostralmiddlefrontal",
            "ctx-rh-rostralmiddlefrontal",
            "Left-Caudate",
            "Right-Caudate",
        ],
        "Sensory": [
            "ctx-lh-postcentral",
            "ctx-rh-postcentral",
            "Left-Thalamus-Proper",
            "Right-Thalamus-Proper",
        ],
        "Visual": [
            "ctx-lh-lateraloccipital",
            "ctx-rh-lateraloccipital",
            "ctx-lh-lingual",
            "ctx-rh-lingual",
            "ctx-lh-pericalcarine",
            "ctx-rh-pericalcarine",
        ],
    }

    def __init__(self, custom_regions: Optional[Dict[str, List[str]]] = None):
        """
        Initialize brain regions.

        Args:
            custom_regions: Optional dictionary of custom region groupings
        """
        self.custom_regions = custom_regions or {}

    def get_structural_regions(
        self, available_columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get structural brain region groupings.

        Args:
            available_columns: Optional list of available column names to filter against

        Returns:
            Dictionary mapping region group names to lists of region names
        """
        regions = {}

        for group_name, region_def in self.FREESURFER_STRUCTURAL.items():
            if isinstance(region_def, str):
                # Pattern-based matching (e.g., "ctx" for cortex)
                if available_columns is not None:
                    pattern = re.compile(f"^{region_def}")
                    regions[group_name] = [
                        col for col in available_columns if pattern.match(col)
                    ]
                else:
                    # Return pattern if no columns provided
                    regions[group_name] = []
            else:
                # Explicit list of regions
                if available_columns is not None:
                    # Filter to only available regions
                    regions[group_name] = [r for r in region_def if r in available_columns]
                else:
                    regions[group_name] = region_def.copy()

        return regions

    def get_functional_regions(
        self, available_columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get functional brain region groupings.

        Args:
            available_columns: Optional list of available column names to filter against

        Returns:
            Dictionary mapping functional group names to lists of region names
        """
        regions = {}

        for group_name, region_list in self.FREESURFER_FUNCTIONAL.items():
            if available_columns is not None:
                # Filter to only available regions
                regions[group_name] = [r for r in region_list if r in available_columns]
            else:
                regions[group_name] = region_list.copy()

        return regions

    def get_custom_regions(
        self, available_columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get custom region groupings.

        Args:
            available_columns: Optional list of available column names to filter against

        Returns:
            Dictionary of custom region groupings
        """
        if available_columns is None:
            return self.custom_regions.copy()

        regions = {}
        for group_name, region_list in self.custom_regions.items():
            regions[group_name] = [r for r in region_list if r in available_columns]

        return regions

    def get_all_regions(
        self, available_columns: Optional[List[str]] = None, grouping: str = "structural"
    ) -> Dict[str, List[str]]:
        """
        Get all region groupings.

        Args:
            available_columns: Optional list of available column names
            grouping: Type of grouping ('structural', 'functional', 'custom', or 'all')

        Returns:
            Dictionary of region groupings
        """
        if grouping == "structural":
            return self.get_structural_regions(available_columns)
        elif grouping == "functional":
            return self.get_functional_regions(available_columns)
        elif grouping == "custom":
            return self.get_custom_regions(available_columns)
        elif grouping == "all":
            # Combine all groupings
            regions = {}
            regions.update(self.get_structural_regions(available_columns))
            regions.update(self.get_functional_regions(available_columns))
            regions.update(self.get_custom_regions(available_columns))
            return regions
        else:
            raise ValueError(
                f"Unknown grouping: {grouping}. "
                f"Must be 'structural', 'functional', 'custom', or 'all'"
            )

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        regions: Union[str, List[str]],
        grouping: str = "structural",
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Filter DataFrame to specific brain regions.

        Args:
            df: Input DataFrame with ROI columns
            regions: Region group name(s) to include, or "all"
            grouping: Type of grouping ('structural', 'functional', 'custom')
            include_metadata: If True, keep non-ROI columns (e.g., age, dose)

        Returns:
            Filtered DataFrame

        Examples:
            >>> # Filter to basal ganglia only
            >>> df_bg = regions.filter_dataframe(df, "BasalGanglia")
            >>>
            >>> # Filter to motor and limbic systems
            >>> df_motor = regions.filter_dataframe(
            ...     df, ["Motor", "Limbic"], grouping="functional"
            ... )
        """
        # Get region groupings
        region_dict = self.get_all_regions(df.columns.tolist(), grouping=grouping)

        # Determine which columns to keep
        if regions == "all":
            # Keep all ROI columns
            roi_columns = []
            for region_list in region_dict.values():
                roi_columns.extend(region_list)
        else:
            # Convert to list if single string
            if isinstance(regions, str):
                regions = [regions]

            # Collect columns for specified regions
            roi_columns = []
            for region_name in regions:
                if region_name not in region_dict:
                    print(f"Warning: Region '{region_name}' not found in {grouping} groupings")
                    continue
                roi_columns.extend(region_dict[region_name])

        # Remove duplicates
        roi_columns = list(set(roi_columns))

        # Identify non-ROI columns (metadata)
        if include_metadata:
            # Assume non-ROI columns are metadata
            # (numeric columns not in any region grouping)
            all_roi_columns = set()
            for region_list in region_dict.values():
                all_roi_columns.update(region_list)

            metadata_columns = [col for col in df.columns if col not in all_roi_columns]
            columns_to_keep = roi_columns + metadata_columns
        else:
            columns_to_keep = roi_columns

        # Filter DataFrame
        missing_cols = set(columns_to_keep) - set(df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} columns not found in DataFrame")

        available_cols = [col for col in columns_to_keep if col in df.columns]

        return df[available_cols]

    @staticmethod
    def get_hemisphere(region_name: str) -> Optional[str]:
        """
        Determine hemisphere from region name.

        Args:
            region_name: FreeSurfer region name

        Returns:
            'left', 'right', 'bilateral', or None
        """
        region_lower = region_name.lower()

        if "left" in region_lower or "lh-" in region_lower:
            return "left"
        elif "right" in region_lower or "rh-" in region_lower:
            return "right"
        elif "bilateral" in region_lower or any(
            x in region_lower for x in ["brain-stem", "cc_", "3rd", "4th", "csf"]
        ):
            return "bilateral"
        else:
            return None

    @staticmethod
    def create_bilateral_pairs(regions: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Create pairs of bilateral regions.

        Args:
            regions: List of region names

        Returns:
            Dictionary mapping base names to left/right region names

        Example:
            >>> regions = ["Left-Thalamus", "Right-Thalamus", "Left-Caudate"]
            >>> pairs = BrainRegions.create_bilateral_pairs(regions)
            >>> # {'Thalamus': {'left': 'Left-Thalamus', 'right': 'Right-Thalamus'}}
        """
        pairs = {}

        for region in regions:
            # Remove hemisphere prefix
            if region.startswith("Left-"):
                base_name = region[5:]  # Remove "Left-"
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]["left"] = region
            elif region.startswith("Right-"):
                base_name = region[6:]  # Remove "Right-"
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]["right"] = region
            elif region.startswith("ctx-lh-"):
                base_name = region[7:]  # Remove "ctx-lh-"
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]["left"] = region
            elif region.startswith("ctx-rh-"):
                base_name = region[7:]  # Remove "ctx-rh-"
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]["right"] = region

        # Filter to only complete pairs
        complete_pairs = {k: v for k, v in pairs.items() if "left" in v and "right" in v}

        return complete_pairs

    def add_custom_region(self, name: str, regions: List[str]):
        """
        Add a custom region grouping.

        Args:
            name: Name of the custom region group
            regions: List of region names to include
        """
        self.custom_regions[name] = regions

    def remove_custom_region(self, name: str):
        """
        Remove a custom region grouping.

        Args:
            name: Name of the custom region group to remove
        """
        if name in self.custom_regions:
            del self.custom_regions[name]


class RegionSelector:
    """Utility for selecting and filtering brain regions."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize region selector.

        Args:
            df: DataFrame with ROI data
        """
        self.df = df
        self.regions = BrainRegions()

    def select_regions(
        self,
        groups: Union[str, List[str]] = "all",
        grouping: str = "structural",
        exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select brain regions from DataFrame.

        Args:
            groups: Region group(s) to include or "all"
            grouping: Type of grouping ('structural', 'functional', 'custom')
            exclude: Optional list of specific regions to exclude

        Returns:
            DataFrame with selected regions
        """
        # Filter to region groups
        df_filtered = self.regions.filter_dataframe(
            self.df, regions=groups, grouping=grouping, include_metadata=True
        )

        # Exclude specific regions if requested
        if exclude is not None:
            cols_to_keep = [col for col in df_filtered.columns if col not in exclude]
            df_filtered = df_filtered[cols_to_keep]

        return df_filtered

    def get_available_groups(self, grouping: str = "structural") -> List[str]:
        """
        Get list of available region groups.

        Args:
            grouping: Type of grouping

        Returns:
            List of available group names
        """
        region_dict = self.regions.get_all_regions(self.df.columns.tolist(), grouping=grouping)
        # Filter to groups that have at least one column in the DataFrame
        available = [name for name, regions in region_dict.items() if len(regions) > 0]
        return available
