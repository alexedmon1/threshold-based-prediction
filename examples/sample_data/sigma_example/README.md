# SIGMA Atlas Example Dataset

This directory contains a complete synthetic example demonstrating the threshold-based prediction pipeline using rat brain MRI data with a SIGMA-like atlas.

## Dataset Overview

**Synthetic Study Design:**
- 20 synthetic rats with brain MRI data (R1 and R2* maps)
- 4 exposure groups: control, low, medium, high
- Exposure variable: exposure_dose (arbitrary units)
- Range: 0 to 60

**Synthetic neurological effects:**
- Exposure effects concentrated in basal ganglia (caudate-putamen, globus pallidus, substantia nigra)
- Increases R1 relaxation rate (appears bright on T1-weighted MRI)
- Decreases R2* (increased susceptibility effects)
- Effects are dose-dependent and region-specific

**MRI Modalities:**
- **R1 maps**: Longitudinal relaxation rate (1/T1), sensitive to certain contrast agents and pathologies
- **R2* maps**: Effective transverse relaxation rate, sensitive to iron and other paramagnetic substances

## Files in This Example

### Input Data

1. **`sigma_example_metadata.csv`**
   - Rat demographics and exposure data
   - Columns: rat_id, exposure_group, exposure_dose, age_days, weight_g, exposure_duration_days

2. **`sigma_atlas.nii.gz`**
   - Simplified SIGMA-like atlas (16 bilateral brain regions)
   - 64×64×32 voxels, 1mm isotropic resolution

3. **`sigma_labels.csv`**
   - Mapping of label IDs to region names
   - Includes hemisphere information

4. **`rats/`**
   - Synthetic MRI data for 20 rats
   - Each rat directory contains:
     - `R1_registered.nii.gz` - R1 relaxation map
     - `R2star_registered.nii.gz` - R2* relaxation map

5. **`sigma_example_config.yaml`**
   - Configuration file for data preparation
   - Specifies atlas, image patterns, statistics, and metadata

### Brain Regions in Simplified Atlas

- **Basal Ganglia** (highly affected by exposure):
  - Caudate-Putamen (striatum)
  - Globus Pallidus
  - Substantia Nigra
- **Thalamus** (moderately affected)
- **Hippocampus**
- **Cortex**:
  - Motor cortex (M1)
  - Somatosensory cortex (S1)
- **Cerebellum**

### Generated Output

6. **`sigma_example_data.csv`**
   - Prepared data ready for analysis
   - 20 rats × 134 features
     - 16 regions × 2 modalities (R1, R2*) × 4 statistics (mean, median, std, variance) = 128 imaging features
     - Plus metadata columns
   - Merged with exposure metadata

7. **`sigma_example_results.csv`**
   - Threshold scan results
   - Columns: threshold, accuracy, n_low, n_high, n_components, explained_variance

8. **`sigma_example_report.html`**
   - Self-contained HTML report with embedded visualizations
   - Includes accuracy plots, confusion matrices, ROC curves, and key thresholds

## Key Results

**Optimal Threshold: 20.0**
- Classification accuracy: 100.0%
- Balanced groups: 10 low-dose vs 10 high-dose
- PCA components: 11 (explaining 90.9% variance)
- Alternative threshold: 30.0 also achieved 100% accuracy

**Interpretation:**
The synthetic data demonstrates perfect separation of rats based on brain MRI patterns at an exposure threshold of 20. This represents the boundary between low/medium exposure groups and suggests this as a critical dose for detecting neurological changes in the basal ganglia.

## Reproducing This Example

### Step 1: Generate Synthetic Data (Already Done)

```bash
cd examples/sample_data/sigma_example
python generate_synthetic_sigma.py
```

### Step 2: Prepare Data

```bash
neurothreshold prepare \
    --config sigma_example_config.yaml \
    --output sigma_example_data.csv
```

### Step 3: Run Threshold Analysis

```bash
neurothreshold analyze \
    --data sigma_example_data.csv \
    --target exposure_dose \
    --threshold-min 0 \
    --threshold-max 60 \
    --threshold-step 10 \
    --output sigma_example_results.csv \
    --report sigma_example_report.html
```

## Using This Example Programmatically

```python
from threshold_prediction.data.pipeline_factory import DataPipelineFactory
from threshold_prediction.models.threshold_analyzer import ThresholdAnalyzer
from threshold_prediction.evaluation import ResultsEvaluator, HTMLReportGenerator, ResultsVisualizer

# Data preparation
pipeline = DataPipelineFactory.from_config_file("sigma_example_config.yaml")
data = pipeline.run(output_path="sigma_example_data.csv")

# Threshold analysis
analyzer = ThresholdAnalyzer()
analyzer.load_data("sigma_example_data.csv")
results = analyzer.scan_thresholds(
    target_variable="exposure_dose",
    threshold_range=(0, 60),
    threshold_step=10
)

# Generate report
evaluator = ResultsEvaluator(analyzer.results)
visualizer = ResultsVisualizer(analyzer.results)
report_gen = HTMLReportGenerator(evaluator, visualizer)
report_gen.generate_html_report("sigma_example_report.html", target_variable="exposure_dose")
```

## Comparing with FreeSurfer Example

| Aspect | FreeSurfer Example | SIGMA Example |
|--------|-------------------|---------------|
| **Species** | Human | Rat |
| **Modality** | Structural MRI (volumes, thickness) | Quantitative MRI (R1, R2*) |
| **Atlas** | FreeSurfer aseg + aparc | SIGMA-like label map |
| **Regions** | ~100 cortical/subcortical | 16 bilateral regions |
| **Features** | 100 (volumes, thickness, area) | 134 (multi-modal statistics) |
| **Exposure metric** | Arbitrary units (0.01-2.2) | Arbitrary units (0-60) |
| **Best threshold** | 0.6 | 20.0 |
| **Best accuracy** | 95% | 100% |

## Notes

- This is **synthetic data** generated for demonstration purposes only
- The simplified SIGMA atlas has 16 regions (vs 156 in full SIGMA atlas)
- Real SIGMA atlas available at: https://www.nitrc.org/projects/sigma_template
- The dose-effect relationship is simulated based on published neurotoxicology literature
- Perfect accuracy (100%) reflects clean synthetic data with clear group separation
- Real-world animal data would show more biological variability

## References

For real SIGMA atlas usage:
- Barrière et al. (2019). "The SIGMA rat brain templates and atlases for multimodal MRI data analysis and visualization." *Nature Communications*.
- NITRC SIGMA project: https://www.nitrc.org/projects/sigma_template

## Next Steps

- See `examples/sample_data/freesurfer_example/` for human neuroimaging example
- Consult main `README.md` for package overview
- Read `docs/` for detailed documentation on methods and usage
- Try with real SIGMA atlas and rat brain MRI data
