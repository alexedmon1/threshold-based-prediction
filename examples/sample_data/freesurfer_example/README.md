# FreeSurfer Example Dataset

This directory contains a complete synthetic example demonstrating the threshold-based prediction pipeline using FreeSurfer neuroimaging data.

## Dataset Overview

**Synthetic Study Design:**
- 20 synthetic subjects with FreeSurfer-processed brain MRI data
- 4 exposure groups: control, low, medium, high
- Exposure variable: Manganese time-weighted average (mn_twa) in µg/m³
- Range: 0.01 to 2.2 µg/m³

**Synthetic neurological effects:**
- Higher manganese exposure is associated with reduced volumes in basal ganglia structures (caudate, putamen, pallidum, thalamus)
- Effects are progressive with exposure level
- Cortical thickness and volume also show dose-dependent reductions

## Files in This Example

### Input Data

1. **`freesurfer_example_metadata.csv`**
   - Subject demographics and exposure data
   - Columns: subject_id, exposure_group, mn_twa, age, total_welding_years

2. **`subjects/`**
   - Synthetic FreeSurfer stats files for 20 subjects
   - Each subject has:
     - `stats/aseg.stats` - Subcortical segmentation volumes
     - `stats/lh.aparc.stats` - Left hemisphere cortical parcellation
     - `stats/rh.aparc.stats` - Right hemisphere cortical parcellation

3. **`freesurfer_example_config.yaml`**
   - Configuration file for data preparation
   - Specifies FreeSurfer subjects directory, measures, atlases, and metadata

### Generated Output

4. **`freesurfer_example_data.csv`**
   - Prepared data ready for analysis
   - 20 subjects × 100 FreeSurfer features
   - Merged with exposure metadata

5. **`freesurfer_example_results.csv`**
   - Threshold scan results
   - Columns: threshold, accuracy, n_low, n_high, n_components, explained_variance

6. **`freesurfer_example_report.html`**
   - Self-contained HTML report with embedded visualizations
   - Includes accuracy plots, confusion matrices, ROC curves, and key thresholds

## Key Results

**Optimal Threshold: 0.6 µg/m³**
- Classification accuracy: 95.0%
- Balanced groups: 10 low-exposure vs 10 high-exposure
- PCA components: 15 (explaining 90.2% variance)

**Interpretation:**
The synthetic data demonstrates that brain imaging patterns can successfully distinguish between subjects with manganese exposure below vs above 0.6 µg/m³, suggesting this as a potential threshold for detecting neurological changes.

## Reproducing This Example

### Step 1: Generate Synthetic Data (Already Done)

```bash
cd examples/sample_data/freesurfer_example
python generate_synthetic_freesurfer.py
```

### Step 2: Prepare Data

```bash
threshold-predict prepare \
    --config freesurfer_example_config.yaml \
    --output freesurfer_example_data.csv
```

### Step 3: Run Threshold Analysis

```bash
threshold-predict analyze \
    --data freesurfer_example_data.csv \
    --target mn_twa \
    --threshold-min 0.0 \
    --threshold-max 2.0 \
    --threshold-step 0.2 \
    --output freesurfer_example_results.csv \
    --report freesurfer_example_report.html
```

## Using This Example Programmatically

```python
from threshold_prediction.data.pipeline_factory import DataPipelineFactory
from threshold_prediction.models.threshold_analyzer import ThresholdAnalyzer
from threshold_prediction.evaluation import ResultsEvaluator, HTMLReportGenerator, ResultsVisualizer

# Data preparation
pipeline = DataPipelineFactory.from_config_file("freesurfer_example_config.yaml")
data = pipeline.run(output_path="freesurfer_example_data.csv")

# Threshold analysis
analyzer = ThresholdAnalyzer()
analyzer.load_data("freesurfer_example_data.csv")
results = analyzer.scan_thresholds(
    target_variable="mn_twa",
    threshold_range=(0.0, 2.0),
    threshold_step=0.2
)

# Generate report
evaluator = ResultsEvaluator(analyzer.results)
visualizer = ResultsVisualizer(analyzer.results)
report_gen = HTMLReportGenerator(evaluator, visualizer)
report_gen.generate_html_report("freesurfer_example_report.html", target_variable="mn_twa")
```

## Notes

- This is **synthetic data** generated for demonstration purposes only
- The exposure-brain relationship is simulated based on known neurotoxicological effects
- Real-world data would show more variability and confounding factors
- The high accuracy (95%) reflects the clean synthetic data generation

## Next Steps

- See `examples/sample_data/sigma_example/` for an animal neuroimaging example using SIGMA atlas
- Consult main `README.md` for package overview
- Read `docs/` for detailed documentation on methods and usage
