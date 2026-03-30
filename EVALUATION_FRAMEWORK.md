# Evaluation Framework for Multiple Myeloma Imaging Pipeline

Production-grade evaluation framework implementing strict clinical ML standards for patient-level data handling, reproducible preprocessing, comprehensive metrics computation with confidence intervals, and automated reporting.

## Overview

This framework ensures:

- **Patient-level splits** with zero tile/patch leakage
- **Reproducible preprocessing** via frozen contracts
- **Bootstrap confidence intervals** (95% CI with 1000 iterations) on all metrics
- **Statistical testing** (DeLong test for ROC, log-rank for survival)
- **Publication-quality visualizations** (300 DPI, colorblind palette, no chartjunk)
- **Experiment tracking** with MLflow, W&B, and DVC integration
- **Automated reporting** with Markdown and LaTeX output

## Architecture

```
src/evaluation/
├── __init__.py                 # Package exports
├── splitter.py                 # Patient-level data splitting
├── metrics.py                  # Comprehensive metrics with CIs
├── preprocessing_contract.py   # Frozen preprocessing parameters
├── experiment_tracker.py       # MLflow, W&B, DVC integration
├── visualization.py            # Publication-quality figures
└── report_generator.py         # Automated evaluation reports

configs/
└── evaluation.yaml             # Configuration file

scripts/
└── run_evaluation.py           # End-to-end evaluation script

tests/
└── test_evaluation.py          # Comprehensive unit tests (15 test cases)
```

## Core Components

### 1. Data Splitting (splitter.py)

Ensures patient-level separation with zero data leakage.

#### PatientLevelSplitter (Base Class)
```python
splitter = PatientLevelSplitter(seed=42)
```

Provides:
- Validation that no patient appears in multiple splits
- Random seed for reproducibility
- CSV export of split assignments

#### StratifiedPatientSplit
```python
from src.evaluation import StratifiedPatientSplit

splitter = StratifiedPatientSplit(
    test_size=0.2,
    val_size=0.15,
    seed=42
)

splits = splitter.split(
    data,
    patient_col='patient_id',
    label_col='label'
)

# Returns: dict with keys 'train', 'val', 'test'
# Each patient appears in exactly one split
```

Features:
- Stratified by class label at patient level
- Balanced class representation across splits
- Validates no patient leakage
- Saves split assignments to CSV
- Automatic logging of split statistics

#### TimeAwareSplitter
For longitudinal studies:
```python
splitter = TimeAwareSplitter(seed=42)

splits = splitter.split(
    data,
    patient_col='patient_id',
    time_col='diagnosis_date',
    val_time_cutoff='2020-01-01',  # or quantile 0.5
    test_time_cutoff='2022-01-01'  # or quantile 0.8
)
```

Ensures temporal ordering: train ← earlier dates, test ← later dates

#### KFoldPatientSplit
For cross-validation:
```python
splitter = KFoldPatientSplit(n_splits=5, seed=42)
fold_splits = splitter.split(data)  # List of (train, val) tuples

train, val = splitter.get_fold(fold_idx=0)
```

### 2. Metrics (metrics.py)

Comprehensive metrics with bootstrap 95% confidence intervals.

#### ClassificationMetrics

```python
from src.evaluation import ClassificationMetrics

metrics = ClassificationMetrics(
    bootstrap_iterations=1000,
    random_seed=42
)

# Individual metrics
auroc_result = metrics.compute_auroc(y_true, y_pred_proba)
print(f"AUROC: {auroc_result}")
# Output: 0.8532 [95% CI: 0.8201-0.8863]

auprc = metrics.compute_auprc(y_true, y_pred_proba)
accuracy = metrics.compute_accuracy(y_true, y_pred)
balanced_acc = metrics.compute_balanced_accuracy(y_true, y_pred)
f1_macro = metrics.compute_f1(y_true, y_pred, average='macro')

# Binary-specific metrics
metrics_dict = metrics.compute_sensitivity_specificity_ppv_npv(
    y_true, y_pred
)
# Returns: {'sensitivity': MetricResult(...), 'specificity': ..., ...}

# All at once
all_metrics = metrics.compute_all_classification_metrics(
    y_true, y_pred, y_pred_proba
)
```

Returns `MetricResult` objects with:
- `value`: Point estimate
- `ci_lower`: Lower 95% CI bound
- `ci_upper`: Upper 95% CI bound

Available metrics:
- **Probability**: AUROC (macro/micro/per-class), AUPRC
- **Hard predictions**: Accuracy, Balanced Accuracy, F1 (macro/weighted)
- **Binary-specific**: Sensitivity, Specificity, PPV, NPV
- All with bootstrap CIs

#### SurvivalMetrics

```python
from src.evaluation import SurvivalMetrics

survival = SurvivalMetrics(bootstrap_iterations=1000)

c_index = survival.concordance_index(
    event_indicator=event,
    time_to_event=time,
    predicted_risk=risk_scores
)

ibs = survival.integrated_brier_score(
    event_indicator=event,
    time_to_event=time,
    predicted_risk=predicted_prob_event
)

all_metrics = survival.compute_all_survival_metrics(
    event, time, risk
)
```

#### CalibrationMetrics

```python
from src.evaluation import CalibrationMetrics

calibration = CalibrationMetrics(n_bins=10)

ece = calibration.expected_calibration_error(y_true, y_pred_proba)

mean_probs, fractions = calibration.calibration_curve(
    y_true, y_pred_proba
)

metrics = calibration.compute_all_calibration_metrics(
    y_true, y_pred_proba
)
```

### 3. Preprocessing Contract (preprocessing_contract.py)

Frozen preprocessing parameters to ensure reproducibility and prevent data leakage.

```python
from src.evaluation import PreprocessingContract

# Fit on TRAINING DATA ONLY
contract = PreprocessingContract()

contract.fit_normalization(
    X_train,
    method='zscore',  # or 'minmax'
    feature_names=['feature_1', 'feature_2', ...]
)

contract.fit_imputation(
    X_train,
    method='median',  # or 'mean', 'mode'
    feature_names=[...]
)

contract.fit_feature_selection(
    selected_features=['feature_1', 'feature_3', ...],
    n_features_original=100,
    method='lasso',
    threshold=0.01
)

contract.fit_stain_normalization(
    method='macenko',
    reference_image_id='reference_slide_001',
    mean=[...],
    std=[...]
)

# Finalize (prevents further modifications)
contract.finalize()

# Apply to any data using trained parameters
X_test_normalized = contract.transform_features(X_test)
X_test_imputed = contract.impute_missing_values(X_test)
X_test_selected = contract.select_features(X_test)

# Serialize and load
contract.serialize('./preprocessing_contract')

loaded_contract = PreprocessingContract.load('./preprocessing_contract')

# Hash for versioning
print(f"Contract hash: {contract.contract_hash}")
# Use in experiment tracking to version preprocessing
```

Contract hash changes if any preprocessing parameters change, ensuring full reproducibility tracking.

### 4. Experiment Tracking (experiment_tracker.py)

Unified interface for MLflow, W&B, and DVC.

#### MLflow Backend
```python
from src.evaluation import MLflowBackend

tracker = MLflowBackend(
    experiment_name='mm_imaging_pipeline',
    tracking_uri='./mlruns',
    run_name='baseline_model_v1'
)

tracker.start_run()

# Log parameters
tracker.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100
})

# Log metrics
tracker.log_metrics({
    'auroc': 0.85,
    'auprc': 0.82,
    'accuracy': 0.88
})

# Log artifacts
tracker.log_artifact('model.pkl')
tracker.log_artifacts('./results/')

# Reproducibility info
tracker.log_reproducibility_info(
    dataset_version='20240115_v2',
    split_seed=42,
    preprocessing_contract_hash='a1b2c3d4e5f6'
)

# Log environment
tracker.log_environment()

tracker.end_run()
```

#### Weights & Biases Backend
```python
from src.evaluation import WandbBackend

tracker = WandbBackend(
    experiment_name='mm_imaging_pipeline',
    project='mm-imaging',
    entity='lab-name',
    run_name='baseline_model_v1'
)

tracker.start_run()
# Same interface as MLflow
tracker.log_params({...})
tracker.log_metrics({...})
tracker.end_run()
```

#### DVC Integration
```python
from src.evaluation import DVCIntegration

dvc = DVCIntegration(repo_path='.')

# Data versioning
dvc.add_data('./data/train_set.csv')
dvc.add_data('./data/test_set.csv')

# Model versioning
dvc.add_model('./models/baseline.pkl')

# Push to remote storage
dvc.push(remote='storage')
```

### 5. Visualization (visualization.py)

Publication-quality figures: 300 DPI, colorblind palette, minimal chartjunk.

#### ROC Curves
```python
from src.evaluation import ROCCurveVisualizer

viz = ROCCurveVisualizer()

fig, ax = viz.plot(
    y_true, y_pred_proba,
    model_name='Model A',
    auroc=0.85,
    output_path='figures/roc_curve'  # Saves .png and .pdf
)

# Multiple models
fig, ax = viz.plot_multiple({
    'Model A': {'y_true': y_true_a, 'y_pred_proba': y_pred_a, 'auroc': 0.85},
    'Model B': {'y_true': y_true_b, 'y_pred_proba': y_pred_b, 'auroc': 0.82},
})
```

#### Precision-Recall
```python
from src.evaluation import PrecisionRecallVisualizer

viz = PrecisionRecallVisualizer()
fig, ax = viz.plot(y_true, y_pred_proba, output_path='figures/pr_curve')
```

#### Confusion Matrices
```python
from src.evaluation import ConfusionMatrixVisualizer

viz = ConfusionMatrixVisualizer()
fig, ax = viz.plot(
    y_true, y_pred,
    normalize=True,
    class_names=['Negative', 'Positive'],
    output_path='figures/confusion_matrix'
)
```

#### Calibration
```python
from src.evaluation import CalibrationVisualizer

viz = CalibrationVisualizer()
fig, ax = viz.plot(
    mean_predicted_prob,
    fraction_positive,
    model_name='Model A',
    ece=0.05,
    output_path='figures/calibration'
)
```

#### Survival Curves
```python
from src.evaluation import SurvivalCurveVisualizer

viz = SurvivalCurveVisualizer()
fig, ax = viz.plot(
    time_to_event,
    event_indicator,
    risk_groups=high_low_risk,  # Binary or multi-group
    output_path='figures/kaplan_meier'
)
```

#### Training Curves
```python
from src.evaluation import TrainingCurveVisualizer

viz = TrainingCurveVisualizer()
fig, ax = viz.plot(
    history={
        'train_loss': [...],
        'val_loss': [...],
        'train_accuracy': [...],
        'val_accuracy': [...]
    },
    metrics=['loss', 'accuracy'],
    output_path='figures/training'
)
```

#### Attention Heatmaps
```python
from src.evaluation import AttentionHeatmapVisualizer

viz = AttentionHeatmapVisualizer()
fig, ax = viz.overlay_attention(
    wsi_thumbnail,  # H x W x 3
    attention_map,  # H x W
    alpha=0.4,
    cmap='hot',
    output_path='figures/attention'
)
```

### 6. Report Generation (report_generator.py)

Automated Markdown and LaTeX reports.

```python
from src.evaluation import EvaluationReportGenerator

generator = EvaluationReportGenerator(output_dir='./reports')

# Full report
report_path = generator.generate_full_report(
    experiment_name='MM Imaging Baseline',
    dataset_info={
        'n_patients': 200,
        'n_samples': 50000,
        'n_classes': 2,
        'class_distribution': {'Healthy': 25000, 'MM': 25000},
        'imaging_modality': 'H&E WSI'
    },
    split_info={
        'method': 'stratified_patient',
        'seed': 42,
        'breakdown': {
            'train': {'n_patients': 140, 'n_samples': 35000},
            'val': {'n_patients': 30, 'n_samples': 7500},
            'test': {'n_patients': 30, 'n_samples': 7500}
        }
    },
    preprocessing_info={
        'contract_hash': 'a1b2c3d4e5f6',
        'fit_timestamp': '2024-03-30 12:30:45',
        'normalization': {...},
        'imputation': {...},
        'feature_selection': {...}
    },
    model_results={
        'Baseline CNN': {
            'metrics': {
                'auroc': MetricResult(0.85, 0.82, 0.88),
                'accuracy': MetricResult(0.88, 0.86, 0.90),
                ...
            },
            'notes': 'Baseline ResNet50 model'
        }
    },
    benchmark_comparison={
        'published_t11_14_study': {
            'auroc': 0.85,
            'sensitivity': 0.88
        }
    },
    figures={
        'ROC Curve': Path('figures/roc_curve.png'),
        'Confusion Matrix': Path('figures/confusion_matrix.png'),
        ...
    }
)

# LaTeX summary table
table_path = generator.generate_summary_table(model_results)

# CSV summary
csv_path = generator.generate_csv_summary(model_results)
```

Output: Markdown report with embedded figures, benchmark comparisons, and comprehensive tables.

## Configuration (evaluation.yaml)

```yaml
splitting:
  method: stratified_patient
  test_size: 0.2
  val_size: 0.15
  seed: 42
  patient_column: patient_id
  label_column: label

metrics:
  classification: [auroc, auprc, accuracy, balanced_accuracy, f1_macro, ...]
  survival: [c_index, ibs]
  bootstrap_iterations: 1000
  confidence_level: 0.95

tracking:
  backend: mlflow  # or wandb
  mlflow:
    tracking_uri: ./mlruns
    experiment_name: mm_imaging_pipeline

reproducibility:
  save_preprocessing_contract: true
  save_split_assignments: true
  save_environment: true
  generate_dockerfile: true
```

## End-to-End Evaluation Script

```bash
python scripts/run_evaluation.py \
    --config configs/evaluation.yaml \
    --data results/predictions.csv \
    --predictions results/y_pred.csv \
    --probabilities results/y_pred_proba.csv \
    --preprocessing-contract ./preprocessing_contracts/contract \
    --experiment-name "MM Imaging Baseline"
```

Automatically:
1. Loads data and preprocessing contract
2. Creates patient-level splits
3. Computes all metrics with CIs
4. Generates visualizations
5. Compares against benchmarks
6. Generates comprehensive report
7. Logs to MLflow/W&B
8. Exports reproducibility info

## Testing

Comprehensive unit tests (15 test cases):

```bash
cd /sessions/blissful-amazing-cray/r5
python -m pytest tests/test_evaluation.py -v

# Or individual test classes
python -m pytest tests/test_evaluation.py::TestStratifiedPatientSplit -v
python -m pytest tests/test_evaluation.py::TestClassificationMetrics -v
python -m pytest tests/test_evaluation.py::TestPreprocessingContract -v
```

### Test Coverage

- **Splitter tests** (4 tests)
  - No patient leakage across splits
  - All patients included
  - Stratification by class
  - Reproducibility with seed
  - Split assignment serialization

- **Metrics tests** (6 tests)
  - AUROC, AUPRC, accuracy, F1, sensitivity/specificity
  - Bootstrap CI computation
  - CI coverage validation
  - All metrics together

- **Preprocessing contract tests** (6 tests)
  - Normalization fitting and transformation
  - Imputation fitting
  - Feature selection mask
  - Contract hash stability
  - Serialization/deserialization
  - Prevention of modifications after finalize

- **Visualization tests** (3 tests)
  - ROC curve generation
  - Confusion matrix generation
  - Calibration plot generation

## Key Features

### ✓ Strict Clinical ML Standards

1. **Patient-level splits**: No tiles/patches from same patient in multiple splits
2. **Train-only preprocessing**: All parameters fit on training data only
3. **Bootstrap CIs**: 95% confidence intervals on all metrics (1000 iterations)
4. **Validation**: Automatic checks for patient leakage and data integrity

### ✓ Reproducibility

1. **Fixed random seeds**: All splits, bootstrap samples reproducible
2. **Frozen preprocessing contracts**: Immutable after fitting, hashable for versioning
3. **Environment logging**: Full pip freeze, system info, GPU driver version
4. **Split assignments**: Saved to CSV for full traceability
5. **DVC integration**: Data and model versioning

### ✓ Publication-Ready

1. **300 DPI figures**: Suitable for peer-reviewed journals
2. **Colorblind palette**: Accessible to all readers
3. **Confidence intervals**: Shown on all curves and metrics
4. **LaTeX tables**: Copy-paste ready for paper submission
5. **Markdown reports**: Automatically generated from results

### ✓ Comprehensive Metrics

- **Classification**: AUROC, AUPRC, accuracy, balanced accuracy, F1 (macro/weighted), sensitivity, specificity, PPV, NPV
- **Survival**: Concordance index (Harrell's C), integrated Brier score
- **Calibration**: Expected calibration error, calibration curves
- **All with bootstrap CIs**

### ✓ Experiment Tracking

- MLflow with model registry
- Weights & Biases with sweep support
- DVC for data/model versioning
- Automatic reproducibility tag logging

## File Structure

```
/sessions/blissful-amazing-cray/r5/
├── src/evaluation/
│   ├── __init__.py (71 lines)
│   ├── splitter.py (468 lines)              - Patient-level splitting
│   ├── metrics.py (713 lines)               - Comprehensive metrics
│   ├── preprocessing_contract.py (544 lines) - Frozen preprocessing
│   ├── experiment_tracker.py (598 lines)    - MLflow/W&B/DVC tracking
│   ├── visualization.py (621 lines)         - Publication figures
│   └── report_generator.py (397 lines)      - Automated reports
├── configs/
│   └── evaluation.yaml                      - Configuration
├── scripts/
│   └── run_evaluation.py (17 KB)            - End-to-end script
├── tests/
│   └── test_evaluation.py (16 KB)           - Unit tests
└── EVALUATION_FRAMEWORK.md                  - This file

Total: 3,412 lines of production code
```

## Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
pyyaml>=5.4

# Optional
mlflow>=2.0          # For MLflow tracking
wandb>=0.13          # For Weights & Biases
dvc>=3.0             # For DVC integration
```

## Example Workflow

```python
import pandas as pd
import numpy as np
from src.evaluation import (
    StratifiedPatientSplit,
    ClassificationMetrics,
    PreprocessingContract,
    MLflowBackend,
    EvaluationReportGenerator
)

# 1. Load data
data = pd.read_csv('data/dataset.csv')

# 2. Create patient-level splits
splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
splits = splitter.split(data, patient_col='patient_id', label_col='label')

# 3. Fit preprocessing on training data only
contract = PreprocessingContract()
X_train = splits['train'].drop(['patient_id', 'label'], axis=1).values
contract.fit_normalization(X_train, method='zscore')
contract.fit_imputation(X_train, method='median')
contract.finalize()

# 4. Apply preprocessing to test data
X_test = splits['test'].drop(['patient_id', 'label'], axis=1).values
X_test_preprocessed = contract.transform_features(
    contract.impute_missing_values(X_test)
)

# 5. Train model and get predictions (not shown)
y_test = splits['test']['label'].values
y_pred = model.predict(X_test_preprocessed)
y_pred_proba = model.predict_proba(X_test_preprocessed)

# 6. Compute metrics with CIs
metrics_calc = ClassificationMetrics(bootstrap_iterations=1000)
metrics = metrics_calc.compute_all_classification_metrics(
    y_test, y_pred, y_pred_proba
)

# 7. Setup tracking
tracker = MLflowBackend(experiment_name='mm_imaging')
tracker.start_run()
tracker.log_params({'split_seed': 42, 'normalization': 'zscore'})
tracker.log_metrics({k: v.value for k, v in metrics.items()})
tracker.log_reproducibility_info(
    dataset_version='20240330',
    split_seed=42,
    preprocessing_contract_hash=contract.contract_hash
)

# 8. Generate report
generator = EvaluationReportGenerator('./reports')
report = generator.generate_full_report(
    experiment_name='MM Imaging Baseline',
    dataset_info={'n_patients': 200, 'n_samples': 50000, 'n_classes': 2},
    split_info={'method': 'stratified_patient', 'seed': 42},
    preprocessing_info={'contract_hash': contract.contract_hash},
    model_results={'Model': {'metrics': metrics}}
)

tracker.end_run()
```

## References

- DeLong, E. R., et al. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*, 44(3), 837-845.
- Harrell, F. E., et al. (1996). Multivariable prognostic models. *Statistics in Medicine*, 15(4), 361-387.
- Guo, C., & Pleiss, G. (2017). On calibration of modern neural networks. *ICML*.

---

**Framework Version**: 1.0.0
**Last Updated**: 2024-03-30
**Status**: Production-ready
