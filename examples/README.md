# Example Configurations and Workflows

This directory contains example configuration files and workflows for the threshold-based prediction package.

## Directory Structure

```
examples/
├── configs/                      # Configuration files
│   ├── data_prep_human.yaml     # FreeSurfer (human) data preparation
│   ├── data_prep_animal_rat.yaml # Animal (rat) with SIGMA atlas
│   ├── data_prep_csv.yaml       # Direct CSV input
│   └── analysis_config.yaml     # Threshold analysis configuration
├── notebooks/                    # Jupyter notebooks (coming soon)
└── sample_data/                 # Example datasets (coming soon)
```

## Quick Start Workflows

### 1. Human Neuroimaging (FreeSurfer)

```bash
# Prepare data from FreeSurfer outputs
threshold-predict prepare \
    --config examples/configs/data_prep_human.yaml \
    --output human_data.csv

# Run threshold analysis
threshold-predict analyze \
    --data human_data.csv \
    --target mn_twa \
    --threshold-min 0 \
    --threshold-max 2.0 \
    --threshold-step 0.1 \
    --report report.html
```

### 2. Animal Neuroimaging (Rat/SIGMA)

```bash
# Prepare data from rat MRI with SIGMA atlas
threshold-predict prepare \
    --config examples/configs/data_prep_animal_rat.yaml \
    --output rat_data.csv

# Run analysis
threshold-predict analyze \
    --data rat_data.csv \
    --target mn_dose \
    --threshold-min 0 \
    --threshold-max 50 \
    --threshold-step 5 \
    --output results.csv \
    --report rat_report.html
```

### 3. Pre-formatted CSV Data

```bash
# Prepare from CSV (minimal processing)
threshold-predict prepare \
    --config examples/configs/data_prep_csv.yaml \
    --output prepared_data.csv

# Run analysis
threshold-predict analyze \
    --data prepared_data.csv \
    --target dose_mg_kg \
    --threshold-min 0 \
    --threshold-max 20 \
    --threshold-step 2 \
    --report csv_report.html
```

## Configuration File Guide

### Data Preparation Configs

Choose the appropriate configuration based on your data type:

**`data_prep_human.yaml`** - For FreeSurfer-processed human MRI
- Specify FreeSurfer `SUBJECTS_DIR`
- List subjects to process
- Select measures (volume, thickness, area)
- Merge with metadata CSV

**`data_prep_animal_rat.yaml`** - For rat brain imaging
- Use SIGMA, Waxholm, or custom atlas
- Specify image file patterns (R1, R2*, T1, etc.)
- Extract statistical measures (mean, median, variance, etc.)
- Merge with treatment/exposure metadata

**`data_prep_csv.yaml`** - For pre-formatted data
- Direct CSV input with ROI measurements
- Optional separate metadata file
- Useful for partial coverage or manual measurements

### Analysis Config

**`analysis_config.yaml`** - Controls threshold scanning
- Target variable for threshold creation
- Threshold range and step size
- PCA and SVM parameters
- Cross-validation method
- Ensemble bagging options
- Brain region selection

## Programmatic Usage

All configurations can also be used directly in Python:

```python
from threshold_prediction.data.pipeline_factory import DataPipelineFactory
from threshold_prediction.models.threshold_analyzer import ThresholdAnalyzer

# Data preparation
pipeline = DataPipelineFactory.from_config_file("examples/configs/data_prep_human.yaml")
data = pipeline.run(output_path="human_data.csv")

# Analysis
analyzer = ThresholdAnalyzer.from_config_file("examples/configs/analysis_config.yaml")
analyzer.load_data("human_data.csv")
results = analyzer.scan_thresholds(
    target_variable="mn_twa",
    threshold_range=(0.0, 2.0),
    threshold_step=0.1
)

# Generate report
from threshold_prediction.evaluation import ResultsEvaluator, HTMLReportGenerator, ResultsVisualizer

evaluator = ResultsEvaluator(analyzer.results)
visualizer = ResultsVisualizer(analyzer.results)
report_gen = HTMLReportGenerator(evaluator, visualizer)
report_gen.generate_html_report("report.html", target_variable="mn_twa")
```

## Customizing Configurations

### Common Modifications

**1. Change threshold range:**
```yaml
analysis:
  threshold_range: [0.0, 5.0]  # Wider range
  threshold_step: 0.5           # Coarser steps
```

**2. Use leave-one-out CV:**
```yaml
model:
  cv_method: "loo"
```

**3. Select specific brain regions:**
```yaml
regions:
  use_regions: ["BasalGanglia", "Cortex", "Hippocampus"]
```

**4. Add more statistical measures:**
```yaml
animal:
  statistics:
    - "mean"
    - "median"
    - "std"
    - "variance"
    - "skew"
    - "kurtosis"
    - "percentile_10"
    - "percentile_25"
    - "percentile_75"
    - "percentile_90"
```

## Validation

Test your configuration before running full analysis:

```bash
# Validate data quality
threshold-predict validate --data prepared_data.csv

# Check configuration syntax (will fail gracefully if invalid)
threshold-predict prepare --config your_config.yaml --output test.csv
```

## Next Steps

- See [`docs/`](../docs/) for detailed documentation
- Check [`notebooks/`](./notebooks/) for interactive tutorials (coming soon)
- Read the main [README.md](../README.md) for package overview

## Support

For issues or questions:
- Check the documentation
- Review example configs
- Open an issue on GitHub
