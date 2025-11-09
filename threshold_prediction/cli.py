"""
Command-line interface for threshold-based prediction.

Thin wrapper around the functional API for convenient command-line usage.
All functionality is also available through direct API calls.
"""

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from threshold_prediction import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """
    NeuroThreshold - Neuroimaging-based Threshold Prediction.

    For detailed help: neurothreshold COMMAND --help
    """
    pass


@main.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Config YAML file')
@click.option('--output', type=click.Path(), required=True, help='Output CSV path')
def prepare(config, output):
    """Prepare neuroimaging data for analysis."""
    from threshold_prediction.data.pipeline_factory import DataPipelineFactory

    click.echo(f"Loading configuration from {config}")
    pipeline = DataPipelineFactory.from_config_file(config)

    click.echo(f"Running {pipeline.config.pipeline.type} pipeline...")
    data = pipeline.run(output_path=output)

    click.echo(f"✓ Success: {len(data)} subjects, {len(data.columns)} features")
    click.echo(f"  Output: {output}")


@main.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Prepared data CSV')
@click.option('--target', required=True, help='Target variable column name')
@click.option('--threshold-min', type=float, required=True, help='Minimum threshold')
@click.option('--threshold-max', type=float, required=True, help='Maximum threshold')
@click.option('--threshold-step', type=float, required=True, help='Threshold step size')
@click.option('--output', type=click.Path(), default='results.csv', help='Results CSV (default: results.csv)')
@click.option('--report', type=click.Path(), help='Optional HTML report path')
def analyze(data, target, threshold_min, threshold_max, threshold_step, output, report):
    """Run threshold-based SVM analysis."""
    from threshold_prediction.models.threshold_analyzer import ThresholdAnalyzer
    from threshold_prediction.utils.config import ModelConfig

    click.echo(f"Loading data from {data}")
    analyzer = ThresholdAnalyzer(ModelConfig())
    analyzer.load_data(data)

    click.echo(f"Scanning thresholds: {threshold_min} to {threshold_max} (step: {threshold_step})")
    results_df = analyzer.scan_thresholds(
        target_variable=target,
        threshold_range=(threshold_min, threshold_max),
        threshold_step=threshold_step
    )

    results_df.to_csv(output, index=False)
    optimal = analyzer.get_optimal_threshold()

    click.echo(f"\n✓ Analysis complete")
    click.echo(f"  Optimal threshold: {optimal['threshold']:.4f}")
    click.echo(f"  Best accuracy: {optimal['accuracy']:.1%}")
    click.echo(f"  Results: {output}")

    if report:
        from threshold_prediction.evaluation import ResultsEvaluator, HTMLReportGenerator, ResultsVisualizer

        click.echo(f"\nGenerating HTML report...")
        evaluator = ResultsEvaluator(analyzer.results)
        visualizer = ResultsVisualizer(analyzer.results)
        report_gen = HTMLReportGenerator(evaluator, visualizer)
        report_gen.generate_html_report(output_path=report, target_variable=target)

        click.echo(f"✓ Report: {report}")


@main.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Data CSV to validate')
@click.option('--missing-threshold', type=float, default=0.1, help='Max missing data proportion')
@click.option('--outlier-std', type=float, default=3.0, help='Std devs for outlier detection')
def validate(data, missing_threshold, outlier_std):
    """Validate data quality."""
    from threshold_prediction.data.validation import DataValidator
    from threshold_prediction.utils.config import ValidationConfig

    click.echo(f"Validating: {data}")
    df = pd.read_csv(data, index_col=0)

    config = ValidationConfig(
        check_missing=True,
        missing_threshold=missing_threshold,
        check_outliers=True,
        outlier_std=outlier_std
    )
    validator = DataValidator(config)
    report = validator.validate(df)

    click.echo("\n" + str(report))

    if report.is_valid:
        click.echo("\n✓ Validation PASSED")
    else:
        click.echo("\n✗ Validation FAILED")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
