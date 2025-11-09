#!/usr/bin/env python3
"""
Generate synthetic FreeSurfer data for testing and demonstration.

This script creates realistic FreeSurfer stats files (aseg.stats, lh.aparc.stats, rh.aparc.stats)
for a cohort of synthetic subjects with varying exposure levels.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# FreeSurfer aseg structure names and typical volume ranges (mm³)
ASEG_STRUCTURES = {
    "Left-Lateral-Ventricle": (7000, 15000),
    "Left-Inf-Lat-Vent": (200, 800),
    "Left-Cerebellum-White-Matter": (12000, 18000),
    "Left-Cerebellum-Cortex": (45000, 60000),
    "Left-Thalamus-Proper": (6000, 9000),
    "Left-Caudate": (3000, 4500),
    "Left-Putamen": (4000, 6000),
    "Left-Pallidum": (1500, 2500),
    "3rd-Ventricle": (800, 1500),
    "4th-Ventricle": (1000, 2000),
    "Brain-Stem": (19000, 25000),
    "Left-Hippocampus": (3500, 4800),
    "Left-Amygdala": (1200, 1800),
    "Left-Accumbens-area": (400, 700),
    "Left-VentralDC": (3000, 4500),
    "Right-Lateral-Ventricle": (7000, 15000),
    "Right-Inf-Lat-Vent": (200, 800),
    "Right-Cerebellum-White-Matter": (12000, 18000),
    "Right-Cerebellum-Cortex": (45000, 60000),
    "Right-Thalamus-Proper": (6000, 9000),
    "Right-Caudate": (3000, 4500),
    "Right-Putamen": (4000, 6000),
    "Right-Pallidum": (1500, 2500),
    "Right-Hippocampus": (3500, 4800),
    "Right-Amygdala": (1200, 1800),
    "Right-Accumbens-area": (400, 700),
    "Right-VentralDC": (3000, 4500),
}

# Cortical parcellation regions and typical values
# (thickness_mm, area_mm2, volume_mm3)
APARC_REGIONS = {
    "bankssts": (2.3, 2.7, 1200, 1800, 2500, 4000),
    "caudalanteriorcingulate": (2.4, 2.9, 800, 1400, 2000, 3500),
    "caudalmiddlefrontal": (2.3, 2.8, 2500, 3500, 6000, 9000),
    "cuneus": (1.8, 2.3, 1500, 2200, 2800, 4500),
    "entorhinal": (2.8, 3.8, 300, 600, 1200, 2200),
    "fusiform": (2.5, 3.1, 4000, 5500, 11000, 16000),
    "inferiorparietal": (2.3, 2.8, 6000, 8500, 14000, 21000),
    "inferiortemporal": (2.5, 3.1, 3800, 5200, 10000, 15000),
    "isthmuscingulate": (2.2, 2.8, 1000, 1600, 2500, 4000),
    "lateraloccipital": (2.0, 2.5, 6000, 8000, 12000, 18000),
    "lateralorbitofrontal": (2.4, 3.0, 2800, 3800, 7000, 11000),
    "lingual": (1.9, 2.4, 3200, 4500, 6500, 10000),
    "medialorbitofrontal": (2.2, 2.8, 2000, 3000, 5000, 8000),
    "middletemporal": (2.6, 3.2, 4000, 5500, 11000, 16000),
    "parahippocampal": (2.4, 3.0, 700, 1200, 2000, 3500),
    "paracentral": (2.2, 2.7, 1800, 2600, 4500, 6500),
    "parsopercularis": (2.3, 2.9, 1800, 2600, 4500, 7000),
    "parsorbitalis": (2.4, 3.0, 800, 1300, 2200, 3500),
    "parstriangularis": (2.2, 2.8, 1500, 2200, 3500, 5500),
    "pericalcarine": (1.4, 1.9, 1500, 2200, 2200, 3500),
    "postcentral": (1.8, 2.3, 5000, 7000, 9500, 14000),
    "posteriorcingulate": (2.3, 2.9, 1400, 2100, 3500, 5500),
    "precentral": (2.3, 2.8, 6000, 8000, 14000, 20000),
    "precuneus": (2.2, 2.7, 4500, 6200, 10000, 15000),
    "rostralanteriorcingulate": (2.6, 3.2, 800, 1400, 2500, 4000),
    "rostralmiddlefrontal": (2.1, 2.7, 6500, 9000, 14000, 21000),
    "superiorfrontal": (2.5, 3.1, 9000, 12000, 23000, 34000),
    "superiorparietal": (2.0, 2.5, 6000, 8500, 12500, 18000),
    "superiortemporal": (2.7, 3.3, 5000, 7000, 14000, 21000),
    "supramarginal": (2.4, 3.0, 4500, 6200, 11000, 16000),
    "frontalpole": (2.6, 3.3, 200, 500, 800, 1500),
    "temporalpole": (3.0, 4.0, 500, 900, 2000, 3500),
    "transversetemporal": (2.2, 2.8, 400, 700, 1000, 1800),
    "insula": (2.9, 3.6, 2500, 3500, 8000, 12000),
}


def generate_aseg_stats(subject_id: str, exposure_level: float, output_dir: Path):
    """Generate synthetic aseg.stats file."""
    stats_dir = output_dir / subject_id / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Exposure effect: higher exposure slightly reduces basal ganglia volumes
    exposure_factor = 1.0 - (exposure_level * 0.15)  # Up to 15% reduction at max exposure

    lines = []
    lines.append("# Title Segmentation Statistics")
    lines.append(f"# generating_program mri_segstats")
    lines.append(f"# cmdline mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats")
    lines.append(f"# sysname  Linux")
    lines.append(f"# hostname synthetic-host")
    lines.append(f"# machine  x86_64")
    lines.append(f"# user     synthetic-user")
    lines.append("#")
    lines.append("# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange")

    for idx, (structure, (vol_min, vol_max)) in enumerate(ASEG_STRUCTURES.items(), start=1):
        # Apply exposure effect to basal ganglia structures
        if any(bg in structure for bg in ["Caudate", "Putamen", "Pallidum", "Thalamus"]):
            vol = np.random.uniform(vol_min * exposure_factor, vol_max * exposure_factor)
        else:
            vol = np.random.uniform(vol_min, vol_max)

        nvoxels = int(vol)
        norm_mean = np.random.uniform(40, 80)
        norm_std = np.random.uniform(5, 15)
        norm_min = norm_mean - np.random.uniform(20, 30)
        norm_max = norm_mean + np.random.uniform(20, 30)
        norm_range = norm_max - norm_min

        lines.append(f"{idx:3d} {idx+1000:5d} {nvoxels:8d} {vol:10.1f} {structure:40s} "
                    f"{norm_mean:6.2f} {norm_std:6.2f} {norm_min:6.2f} {norm_max:6.2f} {norm_range:6.2f}")

    with open(stats_dir / "aseg.stats", 'w') as f:
        f.write('\n'.join(lines))


def generate_aparc_stats(subject_id: str, exposure_level: float, output_dir: Path, hemi: str):
    """Generate synthetic aparc stats file (lh or rh)."""
    stats_dir = output_dir / subject_id / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Exposure effect: slight reduction in cortical thickness and volume
    thickness_factor = 1.0 - (exposure_level * 0.08)
    volume_factor = 1.0 - (exposure_level * 0.12)

    lines = []
    lines.append("# Table of FreeSurfer cortical parcellation anatomical statistics")
    lines.append(f"# hemi {hemi}")
    lines.append(f"# subject {subject_id}")
    lines.append("#")
    lines.append("# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd")

    for region, (thick_min, thick_max, area_min, area_max, vol_min, vol_max) in APARC_REGIONS.items():
        thickness = np.random.uniform(thick_min * thickness_factor, thick_max * thickness_factor)
        area = np.random.uniform(area_min, area_max)
        volume = np.random.uniform(vol_min * volume_factor, vol_max * volume_factor)

        num_vert = int(area / 0.7)  # Approximate vertices from area
        thick_std = np.random.uniform(0.3, 0.6)
        mean_curv = np.random.uniform(0.12, 0.18)
        gaus_curv = np.random.uniform(0.03, 0.08)
        fold_ind = np.random.uniform(10, 50)
        curv_ind = np.random.uniform(0.5, 2.0)

        lines.append(f"{region:35s} {num_vert:6d} {area:7.0f} {volume:7.0f} "
                    f"{thickness:6.3f} {thick_std:5.3f} {mean_curv:7.5f} "
                    f"{gaus_curv:7.5f} {fold_ind:6.0f} {curv_ind:6.1f}")

    with open(stats_dir / f"{hemi}.aparc.stats", 'w') as f:
        f.write('\n'.join(lines))


def generate_metadata(n_subjects: int, output_dir: Path):
    """Generate metadata CSV with exposure information."""
    np.random.seed(42)

    subjects = [f"subject_{i:03d}" for i in range(1, n_subjects + 1)]

    # Create groups with different exposure levels
    n_per_group = n_subjects // 4

    exposure_groups = (
        ["control"] * n_per_group +
        ["low"] * n_per_group +
        ["medium"] * n_per_group +
        ["high"] * (n_subjects - 3 * n_per_group)
    )

    # Manganese time-weighted average (µg/m³)
    mn_twa_ranges = {
        "control": (0.01, 0.05),
        "low": (0.2, 0.6),
        "medium": (0.7, 1.3),
        "high": (1.4, 2.2)
    }

    mn_twa = [np.random.uniform(*mn_twa_ranges[group]) for group in exposure_groups]

    # Normalized exposure (0-1 scale for synthesis)
    exposure_normalized = [(val - 0.01) / (2.2 - 0.01) for val in mn_twa]

    # Age and other demographics
    age = np.random.randint(30, 65, n_subjects)
    years_welding = [
        np.random.uniform(0, 2) if g == "control" else
        np.random.uniform(3, 8) if g == "low" else
        np.random.uniform(9, 15) if g == "medium" else
        np.random.uniform(16, 30)
        for g in exposure_groups
    ]

    metadata = pd.DataFrame({
        "subject_id": subjects,
        "exposure_group": exposure_groups,
        "mn_twa": mn_twa,
        "age": age,
        "total_welding_years": years_welding,
        "exposure_normalized": exposure_normalized  # For synthesis only
    })

    metadata.to_csv(output_dir / "freesurfer_example_metadata.csv", index=False)
    return metadata


def main():
    """Generate complete synthetic FreeSurfer dataset."""
    script_dir = Path(__file__).parent
    subjects_dir = script_dir / "subjects"

    print("Generating synthetic FreeSurfer data...")

    # Generate metadata
    n_subjects = 20
    metadata = generate_metadata(n_subjects, script_dir)
    print(f"✓ Generated metadata for {n_subjects} subjects")

    # Generate FreeSurfer stats for each subject
    for _, row in metadata.iterrows():
        subject_id = row["subject_id"]
        exposure = row["exposure_normalized"]

        # Generate aseg.stats
        generate_aseg_stats(subject_id, exposure, subjects_dir)

        # Generate aparc stats for both hemispheres
        generate_aparc_stats(subject_id, exposure, subjects_dir, "lh")
        generate_aparc_stats(subject_id, exposure, subjects_dir, "rh")

        print(f"✓ Generated stats files for {subject_id}")

    print(f"\n✓ Synthetic FreeSurfer data generation complete!")
    print(f"  Subjects directory: {subjects_dir}")
    print(f"  Metadata: {script_dir / 'freesurfer_example_metadata.csv'}")


if __name__ == "__main__":
    main()
