"""
Registry of pre-configured brain atlases for animal neuroimaging.

Supports common atlases for rats, mice, and non-human primates.
"""

from pathlib import Path
from typing import Dict, List, Optional


class AtlasRegistry:
    """Registry for pre-configured animal brain atlases."""

    # Pre-configured atlases with metadata
    ATLASES = {
        "rat": {
            "waxholm": {
                "full_name": "Waxholm Space Atlas of the Sprague Dawley Rat Brain",
                "version": "v3",
                "species": "Rattus norvegicus (Sprague Dawley)",
                "url": "https://www.nitrc.org/projects/whs-sd-atlas",
                "reference": "Papp et al. (2014). Waxholm Space atlas of the Sprague Dawley rat brain. NeuroImage, 97, 374-386.",
                "doi": "10.1016/j.neuroimage.2014.04.001",
                "description": "High-resolution (39 μm) MRI and DTI atlas with 79 anatomical regions",
                "resolution": "39μm isotropic",
                "num_regions": 79,
                "labels": {},  # Populated when needed
            },
            "sigma": {
                "full_name": "SIGMA - Sprague Dawley Rat Brain Atlas",
                "version": "latest",
                "species": "Rattus norvegicus (Sprague Dawley)",
                "url": "https://www.nitrc.org/projects/sigma_template",
                "reference": "Barriere et al. (2019). The SIGMA rat brain templates and atlases for multimodal MRI data analysis and visualization. Nature Communications, 10(1), 5699.",
                "doi": "10.1038/s41467-019-13575-7",
                "description": "Scalable multimodal atlas with detailed parcellation, optimized for quantitative MRI",
                "resolution": "150μm isotropic",
                "num_regions": 156,
                "modalities": ["T1", "T2", "T2*", "FA", "MD"],
                "labels": {},  # Populated when needed
            },
            "schwarz": {
                "full_name": "Schwarz Rat Brain Atlas",
                "version": "2006",
                "species": "Rattus norvegicus",
                "reference": "Schwarz et al. (2006). A stereotaxic MRI template set for the rat brain with tissue class distribution maps and co-registered anatomical atlas. NeuroImage, 32(2), 538-550.",
                "doi": "10.1016/j.neuroimage.2006.04.214",
                "description": "MRI-based stereotaxic atlas",
                "labels": {},
            },
        },
        "mouse": {
            "allen": {
                "full_name": "Allen Mouse Brain Atlas",
                "version": "CCFv3",
                "species": "Mus musculus (C57BL/6J)",
                "url": "http://atlas.brain-map.org/",
                "reference": "Wang et al. (2020). The Allen Mouse Brain Common Coordinate Framework: A 3D reference atlas. Cell, 181(4), 936-953.",
                "doi": "10.1016/j.cell.2020.04.007",
                "description": "Common Coordinate Framework version 3 with comprehensive parcellation",
                "resolution": "10μm isotropic (available at 10, 25, 50, 100μm)",
                "num_regions": 1328,
                "labels": {},
            },
            "aba": {
                "full_name": "Allen Brain Atlas - Common Coordinate Framework",
                "version": "CCFv3",
                "species": "Mus musculus",
                "url": "http://atlas.brain-map.org/",
                "reference": "Wang et al. (2020)",
                "description": "Alias for Allen Mouse Brain Atlas CCFv3",
                "labels": {},
            },
        },
        "nhp": {
            "civm": {
                "full_name": "CIVM Rhesus Macaque Brain Atlas",
                "species": "Macaca mulatta",
                "url": "https://www.civm.duhs.duke.edu/rhesusatlas/",
                "reference": "Calabrese et al. (2015). A diffusion tensor MRI atlas of the postmortem rhesus macaque brain. NeuroImage, 117, 408-416.",
                "doi": "10.1016/j.neuroimage.2015.05.072",
                "description": "High-resolution DTI-based atlas",
                "resolution": "150μm isotropic",
                "labels": {},
            },
            "inia19": {
                "full_name": "INIA19 Primate Brain Atlas",
                "species": "Macaca mulatta",
                "url": "https://www.nitrc.org/projects/inia19",
                "reference": "Rohlfing et al. (2012). The INIA19 template and NeuroMaps atlas for primate brain image parcellation and spatial normalization. Frontiers in Neuroinformatics, 6, 27.",
                "doi": "10.3389/fninf.2012.00027",
                "description": "Population-averaged atlas with 174 anatomical regions",
                "num_regions": 174,
                "labels": {},
            },
            "d99": {
                "full_name": "D99 Digital Brain Atlas",
                "species": "Macaca mulatta",
                "reference": "Reveley et al. (2017). Three-dimensional digital template atlas of the macaque brain. Cerebral Cortex, 27(9), 4463-4477.",
                "doi": "10.1093/cercor/bhw248",
                "description": "High-resolution template with cortical and subcortical parcellation",
                "labels": {},
            },
        },
    }

    def __init__(self):
        """Initialize atlas registry."""
        self.custom_atlases: Dict[str, Dict] = {}

    def get_atlas(self, species: str, atlas_name: str) -> Dict:
        """
        Get atlas configuration.

        Args:
            species: Species name (e.g., 'rat', 'mouse', 'nhp')
            atlas_name: Atlas name (e.g., 'sigma', 'waxholm', 'allen')

        Returns:
            Dictionary with atlas configuration

        Raises:
            ValueError: If species or atlas not found
        """
        species = species.lower()

        # Check custom atlases first
        custom_key = f"{species}_{atlas_name}"
        if custom_key in self.custom_atlases:
            return self.custom_atlases[custom_key]

        # Check pre-configured atlases
        if species not in self.ATLASES:
            raise ValueError(
                f"Unknown species: {species}. Available: {list(self.ATLASES.keys())}"
            )

        if atlas_name not in self.ATLASES[species]:
            raise ValueError(
                f"Unknown atlas: {atlas_name} for species {species}. "
                f"Available: {list(self.ATLASES[species].keys())}"
            )

        return self.ATLASES[species][atlas_name].copy()

    def list_atlases(self, species: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available atlases.

        Args:
            species: Optional species filter

        Returns:
            Dictionary mapping species to list of atlas names
        """
        if species is not None:
            species = species.lower()
            if species not in self.ATLASES:
                return {}
            return {species: list(self.ATLASES[species].keys())}

        return {sp: list(atlases.keys()) for sp, atlases in self.ATLASES.items()}

    def register_custom_atlas(
        self,
        species: str,
        name: str,
        atlas_path: Path,
        labels: Dict[int, str],
        metadata: Optional[Dict] = None,
    ):
        """
        Register a custom atlas.

        Args:
            species: Species name
            name: Atlas name
            atlas_path: Path to atlas NIfTI file
            labels: Dictionary mapping label IDs to region names
            metadata: Optional metadata dictionary
        """
        species = species.lower()
        custom_key = f"{species}_{name}"

        atlas_config = {
            "full_name": f"Custom {species} atlas: {name}",
            "species": species,
            "path": atlas_path,
            "labels": labels,
            "custom": True,
        }

        if metadata is not None:
            atlas_config.update(metadata)

        self.custom_atlases[custom_key] = atlas_config

    def get_atlas_info(self, species: str, atlas_name: str) -> str:
        """
        Get formatted information about an atlas.

        Args:
            species: Species name
            atlas_name: Atlas name

        Returns:
            Formatted string with atlas information
        """
        atlas = self.get_atlas(species, atlas_name)

        info_lines = [
            f"Atlas: {atlas.get('full_name', atlas_name)}",
            f"Species: {atlas.get('species', species)}",
        ]

        if "version" in atlas:
            info_lines.append(f"Version: {atlas['version']}")

        if "description" in atlas:
            info_lines.append(f"Description: {atlas['description']}")

        if "num_regions" in atlas:
            info_lines.append(f"Number of regions: {atlas['num_regions']}")

        if "resolution" in atlas:
            info_lines.append(f"Resolution: {atlas['resolution']}")

        if "reference" in atlas:
            info_lines.append(f"\nReference: {atlas['reference']}")

        if "doi" in atlas:
            info_lines.append(f"DOI: {atlas['doi']}")

        if "url" in atlas:
            info_lines.append(f"URL: {atlas['url']}")

        return "\n".join(info_lines)


def load_atlas_labels(atlas_name: str, labels_file: Path) -> Dict[int, str]:
    """
    Load label mappings from CSV file.

    Args:
        atlas_name: Name of the atlas
        labels_file: Path to CSV file with columns: label_id, region_name

    Returns:
        Dictionary mapping label IDs to region names
    """
    import pandas as pd

    df = pd.read_csv(labels_file)

    # Validate required columns
    required_cols = ["label_id", "region_name"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Labels CSV must contain columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )

    # Create mapping
    labels = dict(zip(df["label_id"], df["region_name"]))

    return labels


def create_labels_template(output_path: Path, atlas_name: str = "custom"):
    """
    Create a template CSV file for atlas labels.

    Args:
        output_path: Path to save template CSV
        atlas_name: Name of the atlas (for documentation)
    """
    import pandas as pd

    template_data = {
        "label_id": [1, 2, 3, 4, 5],
        "region_name": [
            "region_1",
            "region_2",
            "region_3",
            "region_4",
            "region_5",
        ],
        "structure_group": [
            "cortex",
            "cortex",
            "subcortical",
            "cerebellum",
            "white_matter",
        ],
        "hemisphere": ["left", "right", "left", "bilateral", "bilateral"],
    }

    df = pd.DataFrame(template_data)

    # Add header comment
    with open(output_path, "w") as f:
        f.write(f"# Atlas labels template for: {atlas_name}\n")
        f.write("# Required columns: label_id, region_name\n")
        f.write("# Optional columns: structure_group, hemisphere, parent_region\n")
        f.write("#\n")

    # Append dataframe
    df.to_csv(output_path, mode="a", index=False)
