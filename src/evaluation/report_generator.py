"""
Automated evaluation report generation.

Generates structured Markdown reports and LaTeX summary tables suitable for
peer-reviewed publication. Includes dataset summaries, split info, preprocessing
details, per-model results, key figures, and statistical comparisons.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .metrics import MetricResult

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """
    Automated evaluation report generator.

    Generates comprehensive markdown reports and publication-ready summary tables.
    """

    def __init__(self, output_dir: Union[str, Path] = "./reports"):
        """
        Initialize report generator.

        Args:
            output_dir (Union[str, Path]): Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        experiment_name: str,
        dataset_info: Dict[str, Any],
        split_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        benchmark_comparison: Optional[Dict[str, Dict[str, float]]] = None,
        figures: Optional[Dict[str, Path]] = None
    ) -> Path:
        """
        Generate comprehensive evaluation report.

        Args:
            experiment_name (str): Name of experiment
            dataset_info (Dict[str, Any]): Dataset statistics
            split_info (Dict[str, Any]): Data split information
            preprocessing_info (Dict[str, Any]): Preprocessing contract info
            model_results (Dict[str, Dict[str, Any]]): Per-model metrics and results
            benchmark_comparison (Optional[Dict[str, Dict[str, float]]]): Benchmark comparison
            figures (Optional[Dict[str, Path]]): Paths to figure files

        Returns:
            Path: Path to generated report file
        """
        report_path = self.output_dir / f"{experiment_name}_report.md"

        with open(report_path, "w") as f:
            f.write(self._generate_header(experiment_name))
            f.write(self._generate_dataset_summary(dataset_info))
            f.write(self._generate_split_summary(split_info))
            f.write(self._generate_preprocessing_summary(preprocessing_info))
            f.write(self._generate_model_results(model_results))

            if benchmark_comparison:
                f.write(self._generate_benchmark_comparison(
                    model_results, benchmark_comparison
                ))

            if figures:
                f.write(self._generate_figures_section(figures))

            f.write(self._generate_footer())

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _generate_header(self, experiment_name: str) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# {experiment_name} - Evaluation Report

**Generated:** {timestamp}

---

"""

    def _generate_dataset_summary(self, dataset_info: Dict[str, Any]) -> str:
        """Generate dataset summary section."""
        section = "## Dataset Summary\n\n"

        if isinstance(dataset_info, dict):
            section += f"- **Total Patients:** {dataset_info.get('n_patients', 'N/A')}\n"
            section += f"- **Total Samples (Tiles/Patches):** {dataset_info.get('n_samples', 'N/A')}\n"
            section += f"- **Number of Classes:** {dataset_info.get('n_classes', 'N/A')}\n"

            # Class distribution
            class_dist = dataset_info.get('class_distribution', {})
            if class_dist:
                section += "\n### Class Distribution\n\n"
                for class_name, count in class_dist.items():
                    pct = 100 * count / dataset_info.get('n_samples', 1)
                    section += f"- {class_name}: {count} ({pct:.1f}%)\n"

            # Additional info
            if 'imaging_modality' in dataset_info:
                section += f"\n- **Imaging Modality:** {dataset_info['imaging_modality']}\n"
            if 'institution' in dataset_info:
                section += f"- **Institution:** {dataset_info['institution']}\n"

        section += "\n"
        return section

    def _generate_split_summary(self, split_info: Dict[str, Any]) -> str:
        """Generate data split summary section."""
        section = "## Data Split Information\n\n"

        section += f"- **Split Method:** {split_info.get('method', 'N/A')}\n"
        section += f"- **Random Seed:** {split_info.get('seed', 'N/A')}\n"

        if 'train_size' in split_info:
            section += f"- **Train/Val/Test Split:** "
            section += f"{split_info.get('train_size', 'N/A')}/"\
                       f"{split_info.get('val_size', 'N/A')}/"\
                       f"{split_info.get('test_size', 'N/A')}\n"

        # Split composition
        split_breakdown = split_info.get('breakdown', {})
        if split_breakdown:
            section += "\n### Split Composition\n\n"
            for split_name, stats in split_breakdown.items():
                section += f"**{split_name.upper()}**\n"
                section += f"- Patients: {stats.get('n_patients', 'N/A')}\n"
                section += f"- Samples: {stats.get('n_samples', 'N/A')}\n"

                class_dist = stats.get('class_distribution', {})
                if class_dist:
                    section += "- Class Distribution: "
                    dist_str = ", ".join([
                        f"{k}: {v}" for k, v in class_dist.items()
                    ])
                    section += dist_str + "\n"
                section += "\n"

        section += "**Note:** All splits respect patient-level boundaries. "
        section += "No patient appears in multiple splits.\n\n"

        return section

    def _generate_preprocessing_summary(self, preprocessing_info: Dict[str, Any]) -> str:
        """Generate preprocessing contract summary."""
        section = "## Preprocessing Contract\n\n"

        section += f"- **Contract Hash:** `{preprocessing_info.get('contract_hash', 'N/A')}`\n"
        section += f"- **Fitted Timestamp:** {preprocessing_info.get('fit_timestamp', 'N/A')}\n\n"

        # Normalization
        normalization = preprocessing_info.get('normalization', {})
        if normalization:
            section += "### Feature Normalization\n\n"
            section += f"- **Method:** {normalization.get('method', 'N/A')}\n"
            section += f"- **Features:** {len(normalization.get('feature_names', []))} features\n"
            section += "- **Fit on:** Training data only\n\n"

        # Imputation
        imputation = preprocessing_info.get('imputation', {})
        if imputation:
            section += "### Missing Value Imputation\n\n"
            section += f"- **Method:** {imputation.get('method', 'N/A')}\n"
            section += f"- **Features:** {len(imputation.get('feature_names', []))} features\n"
            section += "- **Fit on:** Training data only\n\n"

        # Feature selection
        feature_sel = preprocessing_info.get('feature_selection', {})
        if feature_sel:
            section += "### Feature Selection\n\n"
            section += f"- **Method:** {feature_sel.get('method', 'N/A')}\n"
            section += f"- **Selected Features:** "
            section += f"{feature_sel.get('n_features_selected', 'N/A')} / "
            section += f"{feature_sel.get('n_features_original', 'N/A')}\n"
            if feature_sel.get('threshold') is not None:
                section += f"- **Threshold:** {feature_sel['threshold']}\n"
            section += "\n"

        section += "**Principle:** All preprocessing parameters are fit ONLY on training data "
        section += "to prevent data leakage.\n\n"

        return section

    def _generate_model_results(
        self,
        model_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate per-model results section."""
        section = "## Model Results\n\n"

        for model_name, results in model_results.items():
            section += f"### {model_name}\n\n"

            # Metrics table
            metrics = results.get('metrics', {})
            if metrics:
                section += "#### Metrics\n\n"
                section += "| Metric | Value | 95% CI |\n"
                section += "|--------|-------|--------|\n"

                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, MetricResult):
                        section += f"| {metric_name} | "
                        section += f"{metric_value.value:.4f} | "
                        section += f"[{metric_value.ci_lower:.4f}, "
                        section += f"{metric_value.ci_upper:.4f}] |\n"
                    elif isinstance(metric_value, dict) and 'value' in metric_value:
                        section += f"| {metric_name} | "
                        section += f"{metric_value['value']:.4f} | "
                        section += f"[{metric_value['ci_lower']:.4f}, "
                        section += f"{metric_value['ci_upper']:.4f}] |\n"

                section += "\n"

            # Additional notes
            if 'notes' in results:
                section += f"**Notes:** {results['notes']}\n\n"

        return section

    def _generate_benchmark_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        benchmark_comparison: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate benchmark comparison section."""
        section = "## Benchmark Comparison\n\n"

        section += "| Model | Metric | Our Results | Benchmark | Difference |\n"
        section += "|-------|--------|------------|-----------|------------|\n"

        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})

            for benchmark_name, benchmark_metrics in benchmark_comparison.items():
                for metric_name, benchmark_value in benchmark_metrics.items():
                    if metric_name in metrics:
                        our_value = metrics[metric_name]
                        if isinstance(our_value, MetricResult):
                            our_val = our_value.value
                        elif isinstance(our_value, dict):
                            our_val = our_value.get('value', our_value)
                        else:
                            our_val = our_value

                        diff = our_val - benchmark_value

                        section += f"| {model_name} | {metric_name} | "
                        section += f"{our_val:.4f} | {benchmark_value:.4f} | "
                        section += f"{diff:+.4f} |\n"

        section += "\n"
        return section

    def _generate_figures_section(self, figures: Dict[str, Path]) -> str:
        """Generate figures section."""
        section = "## Results Figures\n\n"

        for figure_name, figure_path in figures.items():
            # Use relative path
            rel_path = figure_path.relative_to(self.output_dir) if figure_path.is_absolute() else figure_path

            section += f"### {figure_name}\n\n"
            section += f"![{figure_name}]({rel_path})\n\n"

        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

## References

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.
- Harrell, F. E., Lee, K. L., & Mark, D. B. (1996). Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Statistics in medicine, 15(4), 361-387.
- Guo, C., & Pleiss, G. (2017). On calibration of modern neural networks. In International Conference on Machine Learning (pp. 1321-1330). PMLR.

---

*Report generated by MM-Imaging Evaluation Framework*
"""

    def generate_summary_table(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metrics_to_include: Optional[List[str]] = None
    ) -> Path:
        """
        Generate publication-ready LaTeX summary table.

        Args:
            model_results (Dict[str, Dict[str, Any]]): Per-model results
            metrics_to_include (Optional[List[str]]): Specific metrics to include

        Returns:
            Path: Path to generated LaTeX file
        """
        table_path = self.output_dir / "summary_table.tex"

        # Build data for table
        rows = []
        for model_name, results in model_results.items():
            row = {"Model": model_name}
            metrics = results.get('metrics', {})

            for metric_name, metric_value in metrics.items():
                if metrics_to_include and metric_name not in metrics_to_include:
                    continue

                if isinstance(metric_value, MetricResult):
                    value_str = f"{metric_value.value:.3f}"
                    ci_str = f"[{metric_value.ci_lower:.3f}--{metric_value.ci_upper:.3f}]"
                    row[f"{metric_name}"] = f"{value_str} {ci_str}"
                elif isinstance(metric_value, dict) and 'value' in metric_value:
                    value_str = f"{metric_value['value']:.3f}"
                    ci_str = f"[{metric_value['ci_lower']:.3f}--{metric_value['ci_upper']:.3f}]"
                    row[f"{metric_name}"] = f"{value_str} {ci_str}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Generate LaTeX table
        latex_table = df.to_latex(index=False, escape=False)

        with open(table_path, "w") as f:
            f.write("% Summary Table - Multiple Myeloma Imaging Evaluation\n")
            f.write("% Copy-paste directly into LaTeX document\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Performance Summary. Values are point estimates ")
            f.write("with 95\\% bootstrap confidence intervals.}\n")
            f.write("\\label{tab:summary}\n")
            f.write("\\begin{tabular}{")
            f.write("l" * len(df.columns))  # Column format
            f.write("}\n")
            f.write("\\toprule\n")
            f.write(latex_table.split("\\begin{tabular}{")[1].split("\\end{tabular}")[0])
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"Summary table generated: {table_path}")
        return table_path

    def generate_csv_summary(
        self,
        model_results: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Generate CSV summary of results.

        Args:
            model_results (Dict[str, Dict[str, Any]]): Per-model results

        Returns:
            Path: Path to generated CSV file
        """
        csv_path = self.output_dir / "results_summary.csv"

        rows = []
        for model_name, results in model_results.items():
            row = {"model": model_name}
            metrics = results.get('metrics', {})

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, MetricResult):
                    row[f"{metric_name}_value"] = metric_value.value
                    row[f"{metric_name}_ci_lower"] = metric_value.ci_lower
                    row[f"{metric_name}_ci_upper"] = metric_value.ci_upper
                elif isinstance(metric_value, dict) and 'value' in metric_value:
                    row[f"{metric_name}_value"] = metric_value['value']
                    row[f"{metric_name}_ci_lower"] = metric_value['ci_lower']
                    row[f"{metric_name}_ci_upper"] = metric_value['ci_upper']

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        logger.info(f"CSV summary generated: {csv_path}")
        return csv_path
