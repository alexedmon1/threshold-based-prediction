# CSV Data Format for Direct Input

## Overview

If you have pre-formatted ROI data (e.g., from manual tracing, other software, or non-imaging measurements), you can use the CSV pipeline to load your data directly without image processing.

This is particularly useful for:
- Animal studies with partial brain coverage (not whole-brain)
- Manual ROI tracing
- Data from non-standard imaging pipelines
- Pre-processed data from other tools
- Non-imaging biomarker data

## Required Format

### Basic Structure

Your CSV file should have:
1. **Subject ID column**: Unique identifier for each subject
2. **ROI/Feature columns**: One column per measurement
3. **Optional metadata columns**: Age, dose, treatment group, etc.

### Example CSV

```csv
subject_id,striatum_left_volume,striatum_right_volume,hippocampus_volume,age_days,dose_mg_kg,treatment_group
rat_001,12.5,12.8,45.2,90,0.0,control
rat_002,11.2,11.5,43.1,95,5.0,low
rat_003,10.8,11.0,41.5,92,10.0,medium
rat_004,9.5,9.8,39.2,88,20.0,high
```

## Column Naming Conventions

### Recommended Naming

Use descriptive names that indicate:
- Region name
- Laterality (left/right) if applicable
- Measurement type
- Units (optional but recommended)

**Good examples:**
- `striatum_left_volume_mm3`
- `cortex_thickness_mm`
- `basal_ganglia_r1_mean`
- `hippocampus_signal_intensity`

**Avoid:**
- Generic names like `col1`, `roi1`, `measure_a`
- Spaces in column names (use underscores)
- Special characters

### Multiple Statistics per ROI

If you have multiple statistics for each ROI (like the legacy code pattern), use suffixes:

```csv
subject_id,striatum_median,striatum_variance,striatum_skew,globus_pallidus_median,globus_pallidus_variance
rat_001,12.5,2.1,0.5,8.3,1.5
rat_002,11.8,2.3,0.6,7.9,1.6
```

## Metadata

### Required Metadata

At minimum, you need:
- **Subject ID**: Unique identifier
- **Target variable**: The exposure/dose variable for threshold analysis

### Recommended Metadata

Include additional metadata for quality control and interpretation:
- **Age** (age_days, age_weeks)
- **Weight** (weight_g)
- **Sex** (M/F or 0/1)
- **Treatment group** (control, low, high)
- **Exposure metrics** (dose_mg_kg, exposure_duration_days, etc.)
- **Behavioral scores** (if applicable)

### Metadata in Separate File

You can provide metadata in a separate CSV:

**roi_data.csv:**
```csv
subject_id,striatum_volume,hippocampus_volume
rat_001,12.5,45.2
rat_002,11.2,43.1
```

**metadata.csv:**
```csv
subject_id,age_days,dose_mg_kg,treatment_group,sacrifice_day
rat_001,90,0.0,control,60
rat_002,95,5.0,low,60
```

## Configuration Example

### Simple CSV Input

```yaml
pipeline:
  type: "csv"

csv:
  data_path: "data/roi_measurements.csv"
  metadata_path: null  # All data in one file

standardization:
  subject_id_column: "subject_id"
  target_variables:
    - "dose_mg_kg"
    - "treatment_group"

validation:
  check_missing: true
  missing_threshold: 0.1
  check_outliers: true
  outlier_std: 3.0
```

### CSV with Separate Metadata

```yaml
pipeline:
  type: "csv"

csv:
  data_path: "data/roi_measurements.csv"
  metadata_path: "data/animal_metadata.csv"

standardization:
  subject_id_column: "subject_id"
  target_variables:
    - "dose_mg_kg"
```

## Creating a Template

Use the built-in template generator:

```python
from threshold_prediction.data.standardizer import DataStandardizer

# Create a template CSV
DataStandardizer.create_data_template(
    output_path="data_template.csv",
    n_rois=10,
    n_subjects=20,
    include_metadata=True
)
```

Or from command line:
```bash
threshold-predict prepare create-template \
    --output data_template.csv \
    --n-rois 10 \
    --n-subjects 20
```

## Data Quality Requirements

### Missing Data

- Maximum 10% missing data per column (configurable)
- Missing values will be imputed using column mean (default)
- Excessive missing data will trigger validation warnings

### Outliers

- Values beyond ±3 standard deviations flagged as outliers (configurable)
- Outliers are reported but not automatically removed
- Use validation report to investigate suspicious values

### Minimum Requirements

- **Subjects**: At least 5 (recommended: >20 for robust analysis)
- **Features**: At least 2 ROI measurements
- **Target variable**: Must be numeric for threshold scanning

## Example Workflow

### 1. Prepare Your Data

Format your data as CSV following the guidelines above.

### 2. Create Configuration

```yaml
# config.yaml
pipeline:
  type: "csv"

csv:
  data_path: "my_roi_data.csv"

standardization:
  subject_id_column: "animal_id"
  target_variables:
    - "manganese_dose"
```

### 3. Run Data Preparation

```python
from threshold_prediction.data.pipeline_factory import DataPipelineFactory

# Load and validate data
pipeline = DataPipelineFactory.from_config_file("config.yaml")
data = pipeline.run(output_path="analysis_ready.csv")
```

Or from command line:
```bash
threshold-predict prepare --config config.yaml --output analysis_ready.csv
```

### 4. Run Threshold Analysis

Once data is prepared, threshold analysis works the same as with image-based data:

```bash
threshold-predict analyze \
    --data analysis_ready.csv \
    --target manganese_dose \
    --threshold-range 0 50 \
    --threshold-step 5
```

## Tips

### Partial Brain Coverage

If your study only measured specific regions (not whole-brain):
- Include only the regions you measured
- No need to have "complete" anatomical coverage
- The algorithm will work with any number of features

### Non-Standard ROIs

If your ROIs don't correspond to standard atlases:
- Use descriptive custom names
- Document your ROI definitions separately
- Include ROI descriptions in a separate documentation file

### Units

- **Be consistent** with units across all measurements
- Include units in column names when possible
- Document units in your study metadata

### Quality Control

Always review the validation report:
```python
from threshold_prediction.data.validation import DataValidator

validator = DataValidator(config.validation)
report = validator.validate(data)
print(report)
```

## Troubleshooting

### "No subject ID column found"
- Ensure you have a column named `subject_id` or specify a different name
- Column names are case-sensitive

### "Insufficient subjects/features"
- Check minimum requirements (5 subjects, 2 features)
- Verify data loaded correctly

### "Data validation failed"
- Review validation report for specific issues
- Check for excessive missing data or outliers
- Verify target variable is numeric

### "Missing required target variable"
- Ensure target variable column exists in your data
- Check spelling and case sensitivity
