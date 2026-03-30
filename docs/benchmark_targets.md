# Published Benchmark Numbers: Imaging Pathology & Radiomics in Multiple Myeloma

## Executive Summary

This document catalogs all published benchmark metrics relevant to the MM imaging pathology and radiomics pipeline. Metrics are organized by task/modality with source citations for reproducibility and validation purposes.

---

## 1. H&E-Based t(11;14) Detection Benchmarks

### Primary Study: Deep-Learning-Based Prediction of t(11;14) in Multiple Myeloma H&E-Stained Samples (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **AUROC (conclusive cases)** | 0.85 | Based on "conclusive" quality samples; overall cohort performance lower |
| **Sensitivity (conclusive)** | 88% | High sensitivity for identifying positive cases |
| **Specificity (conclusive)** | 83.1% | Balanced specificity; some false positives |
| **Overall Accuracy (conclusive)** | 84.3% | Agreement with FISH/molecular reference standard |
| **Dataset** | H&E-stained bone marrow samples, MM patients | Single center |
| **Reference** | MDPI Cancers (2024) | doi: 10.3390/cancers17111733 |

### Secondary Study: Leveraging AI for Predicting Genetic Alterations in Multiple Myeloma Through Morphological Analysis of Bone Marrow Aspirates (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **AUROC (t(11;14))** | 0.925 | Higher than primary study; suggests methodology matters |
| **Sensitivity (t(11;14))** | 91.67% | MIL architecture with self-supervised learning |
| **Specificity (t(11;14))** | 92.20% | Excellent specificity suggests low false positive rate |
| **Training Data** | 1.4M+ bone marrow cells | Massive scale; cell-level labels available |
| **Architecture** | Multiple Instance Learning with self-supervised pretraining | ResNet50 base embeddings + MIL aggregation |
| **Reference** | Blood (ASH Annual Meeting 2024) | Unpublished/conference presentation |

### Comparative Baseline: Prior Machine Learning Approaches (Stanford-Moore et al., pre-2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **Sensitivity (H&E model)** | 67% | Traditional ML; features hand-engineered |
| **Sensitivity (H&E + IHC)** | 75% | Multimodal baseline |
| **Negative Predictive Value (combined)** | 96% | High NPV for ruling out t(11;14) |
| **Reference** | Earlier work referenced in 2024 studies | Baseline for comparison |

### Implications
- **Recent deep learning (88-92% AUROC)** substantially outperforms prior machine learning (67-75% sensitivity)
- **MIL + self-supervised learning (92.5% AUROC)** outperforms standard supervised CNN (85% AUROC), suggesting weak supervision and large cell-level datasets are beneficial
- **High specificity (92.20%) important for clinical utility:** false positive t(11;14) prediction would unnecessarily escalate therapy

---

## 2. Multimodal Morphology + Flow Cytometry Benchmarks

### Study: Multiparameter Flow Cytometry Quantification of Bone Marrow Plasma Cells at Diagnosis Provides More Prognostic Information Than Morphological Assessment (Haematologica, 2013)

| Metric | Value | Notes |
|--------|-------|-------|
| **Flow Cytometry Sensitivity (clonality)** | 96% | Identifies clonal plasma cells |
| **Morphology Sensitivity (plasma cell identification)** | 84% | Lower than flow; subject bias |
| **Immunohistochemistry (IHC) Sensitivity** | 84% | Comparable to morphology |
| **FISH Sensitivity (cytogenetics)** | 79% | Lower than morphology for cell identification but better for specific translocations |
| **Prognostic Information** | Flow > Morphology | Flow-based quantification more predictive of OS |
| **Reference** | Haematologica (2013) | Landmark study on flow vs morphology |

### Study: Report of the European Myeloma Network on Multiparametric Flow Cytometry in Multiple Myeloma and Related Disorders (Haematologica)

| Metric | Value | Notes |
|--------|-------|-------|
| **Standardized Panel Sensitivity** | Not quantified but high | EMDN consensus on antibody panels improves reproducibility |
| **Clonality Assessment** | Flow is gold standard | Most sensitive for identifying aberrant immunophenotypes |
| **Clinical Utility** | High for diagnosis + MRD | Flow enables both diagnosis and treatment monitoring |
| **Reference** | Haematologica | Standardization effort across European centers |

### Implications
- **Flow cytometry is MORE sensitive (96% vs 84%) than morphology** for clonality assessment; complementary diagnostic modalities
- **No published study directly addresses morphology + flow fusion for genetic prediction (e.g., t(11;14))** — this is a research gap
- **Multimodal integration expected to outperform single modality** but lacks quantitative benchmarks

---

## 3. PET/CT Radiomics in NDMM: Risk Stratification Benchmarks

### Study: A Feasibility Study of [18F] FDG PET/CT Radiomics in Predicting High-Risk Cytogenetic Abnormalities in Multiple Myeloma (2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Decision Tree Model (combined PET+CT)** | AUC 0.89 | Validation cohort; best performance |
| **Clinical Baseline** | AUC 0.74 | Standard risk factors; significantly worse |
| **Sensitivity** | 87% | High sensitivity for high-risk detection |
| **Specificity** | 85% | Balanced specificity |
| **Sample Size** | N=129 NDMM patients | Largest radiomics study in MM to date |
| **ROI Strategy** | PET-visible lesions | Metabolically active regions more predictive than CT-only |
| **Reference** | EJNMMI Research / PMC (2025) | Recent; first large-scale PET/CT radiomics for MM genetics |

### Study: Radiomics Feature Analysis for Survival Prediction in Multiple Myeloma: An Automated PET/CT Approach (2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Harrell's C-Index (radiomics risk score)** | 0.60 | Comparable to clinical staging |
| **Harrell's C-Index (ISS)** | 0.59 | International Staging System baseline |
| **Harrell's C-Index (R-ISS)** | 0.57 | Revised ISS; lower than radiomics |
| **Sample Size** | N=443 NDMM patients | Large prospective cohort |
| **Automated Segmentation** | Spine + remaining skeleton | Whole-body approach |
| **Prognostic Features** | Max axial tumor diameter, gray-level homogeneity | Interpretable radiomics signatures |
| **DeepSurv Non-Linear Model** | Mean C-index 0.657 | Superior to traditional Cox regression with radiomics |
| **Reference** | Computers in Biology and Medicine (2025) | State-of-the-art survival modeling |

### Study: Radiomics-Based Biomarkers for Risk Stratification in Newly Diagnosed Multiple Myeloma (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **Radiomics Risk Score C-Index** | 0.60 | Outcome prediction; comparable to ISS |
| **ISS C-Index** | 0.59 | Standard baseline |
| **R-ISS C-Index** | 0.57 | Often cited but slightly worse than radiomics |
| **Independent Prognostic Features** | Max axial diameter, low gray-level concentration | Directly linked to OS |
| **Reference** | ScienceDirect (2024) | Large outcome study |

### Study: Cluster Analysis of Autoencoder-Extracted FDG PET/CT Features Identifies Multiple Myeloma Patients with Poor Prognosis (Scientific Reports, 2023)

| Metric | Value | Notes |
|--------|-------|-------|
| **Unsupervised Clustering Performance** | Significant prognostic stratification | Autoencoder-derived features identify poor prognosis subgroup |
| **Comparison to Traditional Radiomics** | Superior | Deep learning features more prognostically relevant |
| **Reference** | Scientific Reports / Nature (2023) | Shows unsupervised approach competitive with supervised |

### Study: Bone Marrow Segmentation and Radiomics Analysis of [18F]FDG PET/CT Images for Measurable Residual Disease Assessment (2022)

| Metric | Value | Notes |
|--------|-------|-------|
| **MRD Assessment via Radiomics** | Signature correlates with MRD status | Quantifies post-treatment response |
| **Reference** | Computers in Biology and Medicine (2022) | Post-treatment application; fewer publications than baseline risk |

### Implications
- **Radiomics C-index (0.60-0.657) COMPARABLE to ISS (0.59) and R-ISS (0.57)** — suggests prognostic utility equivalent to clinical staging
- **Radiomics achieves AUROC 0.89 for high-risk cytogenetics** — significantly better than clinical models (AUROC 0.74)
- **DeepSurv (C-index 0.657) outperforms traditional radiomics (C-index 0.60)** — suggests neural network survival modeling captures non-linear relationships
- **PET-visible lesion radiomics more predictive than CT-only** — metabolically active regions reflect tumor biology
- **No prospective randomized trial comparing radiomics-guided treatment to standard care** — clinical impact uncertain

---

## 4. Multiple Instance Learning Benchmarks on Hematological Malignancies

### Study: DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning (CVPR 2022)

| Metric | Value | Dataset |
|--------|-------|---------|
| **CAMELYON-16 AUROC** | 98.60%+ | Substantially above prior ABMIL/TransMIL |
| **TCGA Lung Cancer AUROC** | High performance | Outperforms ABMIL/TransMIL baselines |
| **Reference** | CVPR 2022 / IEEE Xplore | Pseudo-bag approach addresses small sample cohorts |

### Study: TransMIL: Transformer Based Correlated Multiple Instance Learning (NeurIPS 2021)

| Metric | Value | Dataset |
|--------|-------|---------|
| **CAMELYON16 AUROC** | 98.82% | State-of-the-art at 2021 |
| **TCGA-NSCLC AUROC** | High | Spatial modeling improves over ABMIL |
| **TCGA-RCC AUROC** | High | Consistent improvement across cancer types |
| **Reference** | NeurIPS 2021 | Pyramidal Position Encoding enables spatial awareness |

### Study: Attention-based Deep Multiple Instance Learning (Ilse et al., ICML 2018)

| Metric | Value | Dataset |
|--------|-------|---------|
| **Histopathology Dataset 1** | Comparable to best MIL | Baseline method; foundational |
| **Histopathology Dataset 2** | Comparable to best MIL | Proved attention mechanism viability |
| **MNIST-based MIL** | Outperforms prior methods | Synthetic benchmark |
| **Reference** | ICML 2018 / PMLR | Foundational attention-MIL paper |

### Study: Data-Efficient and Weakly Supervised Computational Pathology (CLAM, 2021)

| Metric | Value | Dataset |
|--------|-------|---------|
| **Multi-class Cancer Subtyping** | High accuracy | Tested on TCGA, internal cohorts |
| **Transfer to Independent Cohorts** | Good generalization | Slides, biopsies, smartphone microscopy |
| **Reference** | Nature Biomedical Engineering (2021) | Clustering-constrained attention |

### Study: Enhancing Diagnostic Accuracy of Multiple Myeloma Through ML-Driven Analysis of Hematological Slides (Scientific Reports 2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **t(11;14) AUC** | 0.925 | MIL with 1.4M cell-level training |
| **t(11;14) Sensitivity** | 91.67% | High sensitivity for genetic subtype |
| **t(11;14) Specificity** | 92.20% | Balanced specificity |
| **Cell-level Dataset** | 1.4M+ cells | Massive weakly-labeled cell archive |
| **Architecture** | MIL + self-supervised learning | ResNet50 + attention aggregation |
| **Reference** | Scientific Reports (2024) | First large-scale MIL study on MM |

### Implications
- **CAMELYON16 AUROC 98.60-98.82%** is near-human performance but on controlled, single-modality cancer detection task
- **MIL methods transfer well to hematological malignancies** when cell-level or slide-level training data available
- **MIL benchmarks often cherry-picked datasets** — cross-cohort generalization not always tested (challenge from 2025 "Do MIL Models Transfer?" paper)

---

## 5. Pathology Foundation Model Benchmarks

### Study: UNI (Universal Self-Supervised Pathology Image Embedding, 2024)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Tumor Grade Classification** | State-of-the-art | Outperforms Prov-GigaPath (4/5 tasks) |
| **Immunohistochemical Protein Expression Intensity** | State-of-the-art | 4/5 tasks > Prov-GigaPath |
| **Biomarker Prediction** | Competitive with Prov-GigaPath | On treatment outcome tasks |
| **Training Data** | 200M+ images from 350K+ slides | UNI2; includes H&E and IHC |
| **Reference** | Nature Medicine (2024) | Flagship study on universal embeddings |

### Study: CONCH: Vision-Language Pathology Foundation Model (2024)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Microsatellite Instability (MSI) Prediction** | Balanced Accuracy 0.775-0.778 | External validation on PAIP dataset |
| **14-Task Benchmark Suite** | State-of-the-art | Histology classification, segmentation, retrieval tasks |
| **Image Captioning Performance** | Strong | Vision-language alignment enables text-based retrieval |
| **Training Data** | 1.17M image-caption pairs | Diverse histopathology sources |
| **Reference** | Nature Medicine (2024) | Vision-language pretraining approach |

### Study: Prov-GigaPath: Whole-Slide Foundation Model from Real-World Data (Nature 2024)

| Metric | Value | Task |
|--------|-------|------|
| **State-of-the-art on 25/26 Tasks** | Majority dominance | Comprehensive benchmark |
| **Cancer Subtyping (6 types)** | AUROC ≥90% | Breast, kidney, liver, brain, ovarian, CNS |
| **EGFR Mutation Prediction (TCGA)** | AUROC +23.5% vs second-best | Substantial improvement over REMEDIS |
| **AUPRC Improvement (EGFR)** | +66.4% vs second-best | Precision-recall curve significantly better |
| **Pretraining Data** | 1.3B tiles from 171K slides | Real-world clinical data (Providence Health) |
| **Reference** | Nature (2024) | Largest real-world pretraining dataset |

### Study: Virchow: ViT-Huge Biologically Pretrained Model (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **Disease Detection Performance** | Comparable to UNI | Tile-level and slide-level benchmarks |
| **Biomarker/Treatment Outcome Prediction** | Competitive | Periodic parity with UNI and Prov-GigaPath |
| **Training Data** | 2B tiles from ~1.5M slides | 17 tissue types; DINOv2 self-supervision |
| **Reference** | Nature Biomedical Engineering (2024) | Self-supervised alternative to supervised pretraining |

### Study: TITAN: Multimodal Whole-Slide Foundation Model (2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Multimodal Benchmarks** | State-of-the-art | Image + report + caption fusion |
| **Downstream Task Performance** | Superior to image-only | Semantic grounding from pathology reports |
| **Training Data** | 335K WSIs + 423K captions | Whole-slide level; 1.17M image-caption pairs |
| **Reference** | Nature Medicine (2025) | Newest; multimodal vision-language approach |

### Study: Clinical Benchmark of Public Self-Supervised Pathology Foundation Models (2024)

| Metric | Value | Finding |
|--------|-------|--------|
| **UNI/Prov-GigaPath Average AUC** | Highest ranking | No single model dominates all tasks |
| **CONCH Performance** | Competitive | Vision-language approach holds its own |
| **Virchow Performance** | Competitive | Self-supervision competitive with supervised |
| **Task Heterogeneity** | High | Different models excel on different tasks |
| **Reference** | PMC (2024) | Comprehensive multi-center benchmark |

### Study: Current Pathology Foundation Models are Unrobust to Medical Center Differences (2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Uni2-h Robustness** | Most robust of tested models | Still 5-10% performance drop across centers |
| **Virchow2 Robustness** | Most robust of tested models | Similar to Uni2-h |
| **Average Performance Drop** | 5-15% across models | Multi-center domain shift significant |
| **Reference** | arXiv (2025) | Critical finding on real-world deployment |

### Implications
- **Foundation models achieve near-parity on many benchmarks** — no single model clearly superior across all tasks
- **Domain shift (5-15% performance drop) is MAJOR practical limitation** — not addressed by any published foundation model
- **Vision-language models (CONCH, TITAN) add value** when associated text available (pathology reports)
- **Real-world pretraining (Prov-GigaPath) outperforms research-database pretraining on some tasks** but introduces other biases

---

## 6. Lytic Bone Lesion Detection Benchmarks (CT)

### Study: A Deep Learning Algorithm for Detecting Lytic Bone Lesions of Multiple Myeloma on CT (2022)

| Metric | Value | Notes |
|--------|-------|-------|
| **Lesion-Level Sensitivity** | 91.6% | High sensitivity; detects small lesions |
| **Lesion-Level Specificity** | 84.6% | Some false positives acceptable for screening |
| **Lesion Detection AUROC** | 90.4% | Strong discrimination |
| **Two-Step Pipeline** | U-Net (segmentation) + YOLO (detection) | Bone ROI identified, then lesion detection |
| **Dataset** | 40 whole-body CTs; 5,640 annotated lesions | Limited but realistic sample size |
| **Magnification** | Automated detection at clinical resolution | Practical clinical implementation |
| **Reference** | Skeletal Radiology (2022) | Mayo Clinic study; clinically relevant |

### Implications
- **Deep learning achieves >90% sensitivity for lytic lesion detection** — substantial clinical utility for automating laborious manual review
- **Limited dataset size (N=40 CTs)** suggests external validation needed before clinical deployment
- **Two-step approach (segmentation then detection) more interpretable and modular** than end-to-end models

---

## 7. Stain Normalization Benchmarks

### Study: Enhancing Whole Slide Pathology Foundation Models Through Stain Normalization (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy/AUROC Improvement** | 3-8% across tasks | Varies by model and task |
| **Dataset** | Multiple pathology datasets | Heterogeneous staining across sources |
| **Foundation Model Robustness** | Partial; requires normalization for best performance | Not fully invariant to stain variation |
| **Reference** | arXiv (2024) | Recent study on normalization impact |

### Study: Evaluating Effectiveness of Stain Normalization Techniques in Automated Grading (2023)

| Metric | Value | Notes |
|--------|-------|-------|
| **Macenko vs Reinhard vs Vahadane** | Comparative evaluation | Macenko most common; others comparable |
| **Task: Invasive Ductal Carcinoma Grading** | 3-10% accuracy improvement with normalization | Context-dependent |
| **Reference** | Scientific Reports (2023) | Methodological comparison |

### Implications
- **Stain normalization provides 3-8% accuracy improvement** — modest but measurable benefit
- **Not all models require aggressive normalization** — foundation models more robust than older methods
- **Macenko method standard but not universally optimal** — task-specific selection recommended

---

## 8. Summary Table: All Benchmarks at a Glance

| Task | Best Metric | Value | Model/Method | Sample Size | Reference |
|------|-------------|-------|--------------|-------------|-----------|
| **t(11;14) Detection (H&E)** | AUROC | 0.925 | MIL + SSL | 1.4M cells | Blood 2024 |
| **t(11;14) Detection (H&E)** | Sensitivity | 91.67% | MIL + SSL | Large | Blood 2024 |
| **High-Risk MM Cytogenetics (PET/CT)** | AUC | 0.89 | Decision Tree Radiomics | N=129 | EJNMMI 2025 |
| **OS Prediction (PET/CT)** | C-index | 0.657 | DeepSurv Radiomics | N=443 | Computers Biol 2025 |
| **Lytic Lesion Detection (CT)** | Sensitivity | 91.6% | U-Net + YOLO | 40 CTs | Skeletal Radiol 2022 |
| **Flow Clonality Detection** | Sensitivity | 96% | Multiparametric FCM | Varies | Haematologica 2013 |
| **WSI Classification (CAMELYON16)** | AUROC | 98.82% | TransMIL | 400 WSI | NeurIPS 2021 |
| **Foundation Model (MSI Prediction)** | Balanced Accuracy | 0.778 | CONCH | PAIP validation | Nature Med 2024 |
| **Foundation Model (Subtyping)** | AUROC | ≥0.90 | Prov-GigaPath | 6 cancer types | Nature 2024 |
| **Stain Normalization Impact** | Accuracy Gain | 3-8% | Macenko/others | Multiple | arXiv 2024 |

---

## 9. Research Gaps & Unmet Benchmarks

### Missing Benchmarks (Published but Not Quantified)
1. **Morphology + Flow Cytometry Fusion for Genetic Prediction**
   - No published head-to-head comparison of morphology-only vs. morphology+flow for t(11;14) prediction
   - Expected: multimodal fusion should improve over single modality but magnitude unknown

2. **Radiomics Value-Added Study**
   - No randomized controlled trial comparing radiomics-guided treatment selection to standard ISS/R-ISS
   - Published studies show radiomics C-index comparable to ISS but clinical impact unproven

3. **Cross-Center Validation of t(11;14) Morphology Prediction**
   - All published t(11;14) studies single-center or tightly-clustered
   - Domain shift expected but not quantified

4. **Foundation Model Performance on MM-Specific Tasks**
   - Foundation models evaluated on cancer subtypes but not on MM genetic prediction from morphology
   - Unknown if UNI/CONCH/Virchow embeddings transfer to t(11;14), t(4;14), del17p prediction

5. **MIL Model Generalization to External MM Cohorts**
   - MIL methods (TransMIL, CLAM) benchmarked on lung/breast cancer but not validated on independent MM cohorts
   - Transfer learning gap not quantified for hematologic malignancies

### Urgent Research Directions
- **Cross-center validation of morphology-based genetic prediction** (prospective, multi-site)
- **Prospective randomized trial: radiomics-guided vs. standard MM treatment selection**
- **Unsupervised domain adaptation methods** for pathology foundation models
- **Multimodal fusion benchmarking** (morphology+flow+radiomics+genomics)
- **Cell-level weakly-supervised learning** on larger MM cohorts (>10M cells)

---

## References & Data Accessibility

All benchmarks cited are from published peer-reviewed literature, preprints, or conference proceedings available via:
- PubMed/MEDLINE
- arXiv
- Nature/Nature Medicine/Nature Communications
- IEEE Xplore
- PMLR (Proceedings of Machine Learning Research)
- NeurIPS/ICML/CVPR proceedings

For specific methodology details, code, or dataset access:
- GitHub repositories linked in literature_landscape.md
- Direct author contact recommended for unpublished thresholds and hyperparameter details

