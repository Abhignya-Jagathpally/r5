# Evaluation Framework - Quick Start Guide

## Installation

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml mlflow wandb dvc
```

## 5-Minute Overview

### 1. Patient-Level Data Splitting

```python
from src.evaluation import StratifiedPatientSplit

splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
splits = splitter.split(data, patient_col='patient_id', label_col='label')

# Guarantees: no patient appears in multiple splits
train_set = splits['train']  # 140 patients
val_set = splits['val']      # 30 patients
test_set = splits['test']    # 30 patients
```

### 2. Frozen Preprocessing Contract

```python
from src.evaluation import PreprocessingContract

# Fit on TRAINING data ONLY
contract = PreprocessingContract()
contract.fit_normalization(X_train, method='zscore')
contract.fit_imputation(X_train, method='median')
contract.finalize()

# Apply to any data using frozen parameters
X_val_clean = contract.transform_features(X_val)
X_test_clean = contract.transform_features(X_test)

# Save for reproducibility
contract.serialize('./my_contract')
```

### 3. Metrics with Confidence Intervals

```python
from src.evaluation import ClassificationMetrics

metrics = ClassificationMetrics(bootstrap_iterations=1000)
results = metrics.compute_all_classification_metrics(
    y_test, y_pred, y_pred_proba
)

# Each metric has point estimate + 95% CI
print(results['auroc'])  # 0.8532 [95% CI: 0.8201-0.8863]
print(results['accuracy'])
print(results['sensitivity'])
```

### 4. Publication-Quality Visualizations

```python
from src.evaluation import ROCCurveVisualizer, ConfusionMatrixVisualizer

# ROC curves
roc_viz = ROCCurveVisualizer()
roc_viz.plot(y_test, y_pred_proba, model_name='Model A',
             output_path='figures/roc')  # Saves .png and .pdf

# Confusion matrix
cm_viz = ConfusionMatrixVisualizer()
cm_viz.plot(y_test, y_pred, normalize=True,
            output_path='figures/confusion_matrix')
```

### 5. Experiment Tracking

```python
from src.evaluation import MLflowBackend

tracker = MLflowBackend(experiment_name='mm_imaging')
tracker.start_run()
tracker.log_params({'seed': 42, 'batch_size': 32})
tracker.log_metrics({'auroc': 0.85, 'accuracy': 0.88})
tracker.log_reproducibility_info(
    dataset_version='20240330',
    split_seed=42,
    preprocessing_contract_hash='a1b2c3d4e5f6'
)
tracker.end_run()
```

### 6. Automated Reports

```python
from src.evaluation import EvaluationReportGenerator

generator = EvaluationReportGenerator('./reports')
report = generator.generate_full_report(
    experiment_name='Baseline Model',
    dataset_info={'n_patients': 200, 'n_classes': 2},
    split_info={'method': 'stratified_patient', 'seed': 42},
    preprocessing_info={'contract_hash': 'a1b2c3d4e5f6'},
    model_results={'Model': {'metrics': results}}
)
# Generates: results_report.md, summary_table.tex, results_summary.csv
```

## Key Principles

✓ **Patient-level splits only** - No tile/patch leakage
✓ **Train-only fitting** - All preprocessing fit on training data
✓ **Frozen contracts** - Preprocessing immutable and hashable
✓ **Bootstrap CIs** - 95% confidence intervals on all metrics
✓ **Reproducible** - Fixed seeds, versioned contracts
✓ **Publication-ready** - 300 DPI, colorblind palette, LaTeX tables

## Common Commands

```bash
# Run full evaluation pipeline
python scripts/run_evaluation.py \
    --config configs/evaluation.yaml \
    --data results/data.csv \
    --predictions results/y_pred.csv \
    --probabilities results/y_pred_proba.csv

# Run tests
python -m pytest tests/test_evaluation.py -v

# Check syntax
python -m py_compile src/evaluation/*.py

# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

## File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `splitter.py` | Patient-level splitting | 468 |
| `metrics.py` | Metrics with CIs | 713 |
| `preprocessing_contract.py` | Frozen preprocessing | 544 |
| `experiment_tracker.py` | MLflow/W&B/DVC | 598 |
| `visualization.py` | Publication figures | 621 |
| `report_generator.py` | Automated reports | 397 |

## Metrics Available

**Classification**: AUROC, AUPRC, Accuracy, Balanced Accuracy, F1 (macro/weighted), Sensitivity, Specificity, PPV, NPV

**Survival**: Concordance Index, Integrated Brier Score

**Calibration**: Expected Calibration Error, Calibration Curves

All with 95% bootstrap confidence intervals (1000 iterations).

## Visualizations Available

- ROC Curves (single and multi-model)
- Precision-Recall Curves
- Confusion Matrices (normalized and raw)
- Calibration Plots
- Kaplan-Meier Survival Curves
- Training Curves
- Feature Importance
- Attention Heatmaps on WSI

All: 300 DPI, colorblind palette, PNG + PDF formats

---

**Ready to use. No additional setup required.**
