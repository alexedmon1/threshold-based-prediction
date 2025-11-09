#!/usr/bin/env python3
"""
Generate synthetic rat brain MRI data with SIGMA atlas for testing and demonstration.

This script creates realistic NIfTI images (R1, R2*) and a simplified SIGMA-like atlas
for a cohort of synthetic rats with varying manganese exposure levels.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path


# Simplified SIGMA atlas regions (subset of full 156 regions)
# Label ID: (Region Name, Baseline R1 ms, Baseline R2* ms)
SIGMA_REGIONS = {
    1: ("Caud_Put_left", 1450, 25),      # Caudate-Putamen (very sensitive to Mn)
    2: ("Caud_Put_right", 1450, 25),
    3: ("Glob_Pall_left", 1380, 22),     # Globus Pallidus (very sensitive to Mn)
    4: ("Glob_Pall_right", 1380, 22),
    5: ("Thalamus_left", 1420, 24),      # Thalamus (moderately sensitive)
    6: ("Thalamus_right", 1420, 24),
    7: ("Hippocampus_left", 1580, 28),   # Hippocampus
    8: ("Hippocampus_right", 1580, 28),
    9: ("Cortex_M1_left", 1650, 30),     # Motor cortex
    10: ("Cortex_M1_right", 1650, 30),
    11: ("Cortex_S1_left", 1680, 31),    # Somatosensory cortex
    12: ("Cortex_S1_right", 1680, 31),
    13: ("Substantia_Nigra_left", 1200, 18),   # Substantia Nigra (very sensitive)
    14: ("Substantia_Nigra_right", 1200, 18),
    15: ("Cerebellum_left", 1550, 27),   # Cerebellum
    16: ("Cerebellum_right", 1550, 27),
    0: ("Background", 0, 0),
}


def create_synthetic_atlas(output_path: Path, shape=(64, 64, 32)):
    """
    Create a simplified synthetic SIGMA-like atlas.

    This creates a 3D label map with simplified bilateral brain regions.
    """
    atlas_data = np.zeros(shape, dtype=np.int16)

    # Define approximate region positions (simplified rat brain geometry)
    z_mid, y_mid, x_mid = shape[2]//2, shape[1]//2, shape[0]//2

    # Striatum (Caudate-Putamen) - bilateral, anterior-dorsal
    atlas_data[x_mid-8:x_mid-2, y_mid+5:y_mid+12, z_mid-5:z_mid+8] = 1  # left
    atlas_data[x_mid+2:x_mid+8, y_mid+5:y_mid+12, z_mid-5:z_mid+8] = 2  # right

    # Globus Pallidus - bilateral, medial to striatum
    atlas_data[x_mid-7:x_mid-4, y_mid+6:y_mid+10, z_mid-3:z_mid+5] = 3  # left
    atlas_data[x_mid+4:x_mid+7, y_mid+6:y_mid+10, z_mid-3:z_mid+5] = 4  # right

    # Thalamus - bilateral, central-posterior
    atlas_data[x_mid-6:x_mid-1, y_mid-2:y_mid+4, z_mid-4:z_mid+6] = 5  # left
    atlas_data[x_mid+1:x_mid+6, y_mid-2:y_mid+4, z_mid-4:z_mid+6] = 6  # right

    # Hippocampus - bilateral, ventral-posterior
    atlas_data[x_mid-8:x_mid-3, y_mid-8:y_mid-3, z_mid-3:z_mid+7] = 7  # left
    atlas_data[x_mid+3:x_mid+8, y_mid-8:y_mid-3, z_mid-3:z_mid+7] = 8  # right

    # Motor cortex - bilateral, dorsal-lateral
    atlas_data[x_mid-15:x_mid-9, y_mid+8:y_mid+15, z_mid-6:z_mid+10] = 9  # left
    atlas_data[x_mid+9:x_mid+15, y_mid+8:y_mid+15, z_mid-6:z_mid+10] = 10  # right

    # Somatosensory cortex - bilateral, dorsal
    atlas_data[x_mid-12:x_mid-6, y_mid+2:y_mid+8, z_mid-8:z_mid+8] = 11  # left
    atlas_data[x_mid+6:x_mid+12, y_mid+2:y_mid+8, z_mid-8:z_mid+8] = 12  # right

    # Substantia Nigra - bilateral, small ventral regions
    atlas_data[x_mid-5:x_mid-2, y_mid-6:y_mid-3, z_mid:z_mid+3] = 13  # left
    atlas_data[x_mid+2:x_mid+5, y_mid-6:y_mid-3, z_mid:z_mid+3] = 14  # right

    # Cerebellum - bilateral, posterior
    atlas_data[x_mid-10:x_mid-2, y_mid-12:y_mid-5, z_mid-10:z_mid+2] = 15  # left
    atlas_data[x_mid+2:x_mid+10, y_mid-12:y_mid-5, z_mid-10:z_mid+2] = 16  # right

    # Create NIfTI image with 1mm isotropic voxels
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(atlas_data, affine)
    nib.save(nifti_img, output_path)

    return atlas_data.shape


def generate_brain_image(atlas_data, region_values, noise_level=0.05, shape=(64, 64, 32)):
    """
    Generate a synthetic brain MRI image based on atlas and region values.

    Parameters:
    - atlas_data: 3D array with region labels
    - region_values: dict mapping label ID to intensity value
    - noise_level: Gaussian noise level (relative to signal)
    """
    image_data = np.zeros(shape, dtype=np.float32)

    # Fill each region with its characteristic value
    for label_id, value in region_values.items():
        mask = (atlas_data == label_id)
        # Add some within-region variability
        region_noise = np.random.normal(0, value * 0.03, mask.sum())
        image_data[mask] = value + region_noise

    # Add global noise
    noise = np.random.normal(0, np.mean(list(region_values.values())) * noise_level, shape)
    image_data += noise

    # Ensure non-negative
    image_data = np.maximum(image_data, 0)

    return image_data


def generate_rat_images(rat_id: str, exposure_level: float, atlas_data, output_dir: Path):
    """
    Generate R1 and R2* maps for a single rat.

    Manganese exposure increases R1 (brightening on T1) and decreases R2* in basal ganglia.
    """
    rat_dir = output_dir / rat_id
    rat_dir.mkdir(parents=True, exist_ok=True)

    # Exposure effects (dose-dependent):
    # - Mn accumulates primarily in basal ganglia (caudate-putamen, globus pallidus, substantia nigra)
    # - Increases R1 relaxation rate (decreases T1, appears bright on T1-weighted)
    # - Decreases R2* (increases susceptibility effects)

    # Calculate exposure-dependent changes
    # exposure_level ranges from 0 (control) to 1 (high exposure)

    r1_values = {}
    r2star_values = {}

    for label_id, (region_name, baseline_r1, baseline_r2star) in SIGMA_REGIONS.items():
        if label_id == 0:  # Background
            r1_values[0] = 0
            r2star_values[0] = 0
            continue

        # Determine sensitivity to Mn based on region
        if "Caud_Put" in region_name or "Glob_Pall" in region_name or "Substantia_Nigra" in region_name:
            # Highly sensitive regions
            r1_change = exposure_level * 0.35  # Up to 35% increase
            r2star_change = exposure_level * -0.30  # Up to 30% decrease
        elif "Thalamus" in region_name:
            # Moderately sensitive
            r1_change = exposure_level * 0.15
            r2star_change = exposure_level * -0.12
        else:
            # Less sensitive regions
            r1_change = exposure_level * 0.05
            r2star_change = exposure_level * -0.04

        # Apply changes with some random variation
        r1_values[label_id] = baseline_r1 * (1 + r1_change + np.random.normal(0, 0.02))
        r2star_values[label_id] = baseline_r2star * (1 + r2star_change + np.random.normal(0, 0.02))

    # Generate images
    r1_data = generate_brain_image(atlas_data, r1_values, noise_level=0.04)
    r2star_data = generate_brain_image(atlas_data, r2star_values, noise_level=0.05)

    # Save as NIfTI
    affine = np.eye(4)
    r1_img = nib.Nifti1Image(r1_data, affine)
    r2star_img = nib.Nifti1Image(r2star_data, affine)

    nib.save(r1_img, rat_dir / "R1_registered.nii.gz")
    nib.save(r2star_img, rat_dir / "R2star_registered.nii.gz")


def generate_metadata(n_rats: int, output_dir: Path):
    """Generate metadata CSV with manganese exposure information."""
    np.random.seed(123)  # Different seed than FreeSurfer example

    rats = [f"rat_{i:03d}" for i in range(1, n_rats + 1)]

    # Create groups with different exposure levels
    n_per_group = n_rats // 4

    exposure_groups = (
        ["control"] * n_per_group +
        ["low"] * n_per_group +
        ["medium"] * n_per_group +
        ["high"] * (n_rats - 3 * n_per_group)
    )

    # Manganese dose (mg/kg body weight)
    mn_dose_ranges = {
        "control": (0, 0),
        "low": (5, 15),
        "medium": (20, 35),
        "high": (40, 60)
    }

    mn_dose = [
        np.random.uniform(*mn_dose_ranges[group]) if group != "control"
        else 0
        for group in exposure_groups
    ]

    # Normalized exposure (0-1 scale for synthesis)
    exposure_normalized = [dose / 60.0 for dose in mn_dose]

    # Other variables
    age_days = np.random.randint(60, 120, n_rats)
    weight_g = np.random.uniform(250, 350, n_rats)
    exposure_duration_days = [
        0 if g == "control" else np.random.randint(7, 14) if g == "low"
        else np.random.randint(14, 28) if g == "medium"
        else np.random.randint(28, 56)
        for g in exposure_groups
    ]

    metadata = pd.DataFrame({
        "rat_id": rats,
        "exposure_group": exposure_groups,
        "mn_dose": mn_dose,
        "age_days": age_days,
        "weight_g": weight_g,
        "exposure_duration_days": exposure_duration_days,
        "exposure_normalized": exposure_normalized  # For synthesis only
    })

    metadata.to_csv(output_dir / "sigma_example_metadata.csv", index=False)
    return metadata


def generate_labels_csv(output_dir: Path):
    """Generate labels CSV mapping label IDs to region names."""
    labels_data = [
        {"label_id": label_id, "region_name": name, "hemisphere": "left" if "left" in name else "right" if "right" in name else "bilateral"}
        for label_id, (name, _, _) in SIGMA_REGIONS.items()
        if label_id != 0
    ]

    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(output_dir / "sigma_labels.csv", index=False)
    return labels_df


def main():
    """Generate complete synthetic SIGMA dataset."""
    script_dir = Path(__file__).parent
    rats_dir = script_dir / "rats"

    print("Generating synthetic SIGMA/rat brain data...")

    # Generate simplified SIGMA atlas
    atlas_path = script_dir / "sigma_atlas.nii.gz"
    shape = create_synthetic_atlas(atlas_path)
    print(f"✓ Created synthetic SIGMA atlas: {atlas_path}")

    # Load atlas for image generation
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.int16)

    # Generate labels CSV
    labels_df = generate_labels_csv(script_dir)
    print(f"✓ Generated labels CSV with {len(labels_df)} regions")

    # Generate metadata
    n_rats = 20
    metadata = generate_metadata(n_rats, script_dir)
    print(f"✓ Generated metadata for {n_rats} rats")

    # Generate MRI images for each rat
    for _, row in metadata.iterrows():
        rat_id = row["rat_id"]
        exposure = row["exposure_normalized"]

        generate_rat_images(rat_id, exposure, atlas_data, rats_dir)
        print(f"✓ Generated R1 and R2* maps for {rat_id}")

    print(f"\n✓ Synthetic SIGMA/rat data generation complete!")
    print(f"  Atlas: {atlas_path}")
    print(f"  Labels: {script_dir / 'sigma_labels.csv'}")
    print(f"  Rats directory: {rats_dir}")
    print(f"  Metadata: {script_dir / 'sigma_example_metadata.csv'}")


if __name__ == "__main__":
    main()
