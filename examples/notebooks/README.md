# Example Notebooks

This directory contains Jupyter notebooks demonstrating the Python API usage for the threshold-based prediction package.

## Available Notebooks

### 1. FreeSurfer API Example
**File**: `freesurfer_api_example.ipynb`

Demonstrates the complete workflow for human neuroimaging data:
- Loading FreeSurfer outputs (aseg.stats, aparc.stats)
- Merging with exposure metadata
- Running threshold-based SVM analysis
- Evaluating results and identifying key thresholds
- Generating visualizations and HTML reports

**Dataset**: 20 synthetic human subjects with FreeSurfer data

### 2. SIGMA Atlas API Example
**File**: `sigma_api_example.ipynb`

Demonstrates the complete workflow for animal neuroimaging data:
- Extracting ROI statistics from quantitative MRI (R1, R2*)
- Using SIGMA atlas for brain parcellation
- Multi-modal, multi-statistic feature extraction
- Running threshold-based SVM analysis
- Generating comprehensive reports

**Dataset**: 20 synthetic rats with R1/R2* MRI maps

## Getting Started

### Prerequisites

1. Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

2. Ensure Jupyter is installed:
```bash
pip install jupyter ipykernel
```

### Running the Notebooks

1. Navigate to the notebooks directory:
```bash
cd examples/notebooks
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open either notebook in your browser and run the cells sequentially

## What You'll Learn

These notebooks demonstrate:

✅ **Data Preparation**
- Loading neuroimaging data from different sources
- Merging with separate metadata CSV files
- Validating and standardizing data

✅ **Threshold Analysis**
- Configuring the analyzer
- Scanning multiple threshold values
- Finding optimal classification thresholds

✅ **Results Evaluation**
- Calculating comprehensive metrics
- Detecting inflection points
- Identifying key thresholds

✅ **Visualization**
- Accuracy vs threshold plots
- Confusion matrices
- Multi-metric comparisons

✅ **Report Generation**
- Creating self-contained HTML reports
- Embedding figures and tables
- Professional styling

## Python API vs CLI

These notebooks focus on the **Python API** for programmatic usage. If you prefer command-line tools, see:
- `examples/sample_data/freesurfer_example/README.md` for CLI examples
- `examples/sample_data/sigma_example/README.md` for CLI examples

## Customization

You can easily adapt these notebooks for your own data:

1. **Change input paths**: Point to your FreeSurfer SUBJECTS_DIR or MRI images
2. **Modify threshold ranges**: Adjust min, max, and step values
3. **Select brain regions**: Filter to specific ROIs
4. **Change cross-validation**: Switch between k-fold and leave-one-out
5. **Try different target variables**: Use any continuous exposure variable

## Output Files

Running these notebooks will create:
- `api_prepared_data.csv` - Combined imaging + metadata
- `api_report.html` - Comprehensive HTML report with visualizations

These are created in the respective example directories and can be safely deleted after inspection.

## Troubleshooting

**Kernel Issues**: If the kernel doesn't start, ensure you're using the correct Python environment:
```bash
python -m ipykernel install --user --name=threshold-prediction
```

**Import Errors**: If modules aren't found, install the package in editable mode:
```bash
cd ../../  # Go to project root
pip install -e .
```

**Path Issues**: Notebooks assume they're run from the `examples/notebooks/` directory. If running from elsewhere, adjust the `example_dir` path in the first code cell.

## Next Steps

After exploring these notebooks:
- Try with your own neuroimaging data
- Experiment with different ML configurations
- Explore the ensemble bagging features
- Read the full documentation in `docs/`
