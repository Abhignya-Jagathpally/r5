# Imaging Pathology & Radiomics Surrogate-Genetics Pipeline for Multiple Myeloma: Literature Review & Landscape Mapping

## Documents Overview

This directory contains comprehensive publication-quality literature review and landscape mapping for the MM imaging pathology and radiomics pipeline.

### Files

#### 1. `literature_landscape.md` (847 lines, 57 KB)
**Comprehensive literature review covering six core research areas:**
- **Section A: Paper Catalog** — 40+ papers across 7 topics with author, year, venue, core claims, datasets, and key metrics
- **Section B: Paper Clusters** — 7 major clusters of shared assumptions and methodologies
- **Section C: Contradiction Table** — 8 direct contradictions between papers with explanations
- **Section D: Top 3 Most-Cited Concepts** — Intellectual lineage (ABMIL, Foundation Models, Morphology-to-Molecular)
- **Section E: Five Unanswered Research Questions** — Critical gaps with methodology to close them
- **Section F: Methodology Comparison** — Analysis of dominant vs. underused research approaches
- **Section G: 400-Word Synthesis** — What the field believes, contests, and has proven
- **Section H: Untested Assumptions** — 7 shared but unvalidated assumptions
- **Section I: Structured Knowledge Map** — Central claim, 5 supporting pillars, 3 contested zones, 2 frontiers
- **Section J: 5-Minute Explainer** — Non-technical summary for executive/clinician audience

**Topics Covered:**
1. WSI pathology in MM (tiling, stain normalization, patch-based classification)
2. Multiple Instance Learning (ABMIL, CLAM, TransMIL, DTFD-MIL) in hematological malignancies
3. Radiomics in MM (PET/CT, bone lesion segmentation, risk stratification)
4. Pathology foundation models (UNI, CONCH, Virchow, TITAN, Prov-GigaPath)
5. H&E-based t(11;14) molecular surrogate prediction
6. Multimodal fusion (morphology + flow cytometry + genomics + imaging)

#### 2. `benchmark_targets.md` (380 lines, 22 KB)
**Structured reference table of all published benchmark metrics:**
- **t(11;14) Detection:** AUROC 0.85-0.925, sensitivity 88-91.67%, specificity 83.1-92.20%
- **PET/CT Radiomics:** High-risk prediction AUC 0.89, OS C-index 0.60-0.657
- **Flow Cytometry:** Clonality detection sensitivity 96% vs morphology 84%
- **MIL Models:** CAMELYON16 AUROC 98.82% (TransMIL), DTFD-MIL state-of-the-art
- **Foundation Models:** Prov-GigaPath achieves AUROC ≥0.90 on 6 cancer subtypes, but 5-15% domain shift to new medical centers
- **Lytic Lesion Detection:** 91.6% sensitivity, 84.6% specificity, AUROC 0.904

**Sections:**
1. H&E-based t(11;14) detection (primary vs. secondary vs. comparative studies)
2. Multimodal morphology+flow (sensitivity comparisons)
3. PET/CT radiomics in NDMM (risk stratification, survival, MRD)
4. MIL benchmarks on hematological malignancies
5. Pathology foundation model performance
6. Lytic bone lesion detection (CT)
7. Stain normalization impact (3-8% accuracy gain)
8. Summary table across all tasks
9. Research gaps & unmet benchmarks

#### 3. `dataset_catalog.md` (534 lines, 20 KB)
**Complete catalog of public datasets with access instructions:**

**Histopathology/Microscopy:**
- **CMB-MML (TCIA):** 1000+ MM patients, radiology + histopathology, genomics via dbGaP, CC-BY 4.0
- **MiMM_SBILab (TCIA):** 85 BM aspirate images, Jenner-Giemsa, 2560×1920 BMP, stain normalization benchmark
- **SN-AM (TCIA):** 190 images (100 MM, 90 B-ALL), nucleus/background masks, intentional stain variability
- **PCMMD (Nature Scientific Data 2025):** 5000+ plasma cells, expert morphologic labels, recent release
- **LabIA-UFBA MM-Dataset (GitHub):** BM aspirates, smartphone microscopy, open access
- **SegPC-2021:** 775 images with segmentation masks, two different capture systems

**Radiomics (PET/CT):**
- **CMB-MML Radiology Component:** Full-body PET/CT for 1000+ MM patients, integrated with genomics

**Summary Table:** Dataset comparison (sample count, annotations, access, licensing)

**Quick-Start Guide:** 3 scenarios (t(11;14) detector, stain normalization, multimodal model) with step-by-step instructions

---

## Key Findings & Insights

### What the Field Collectively Believes
1. **Deep learning extracts diagnostic information from histopathology** via attention-based MIL and foundation models
2. **H&E morphology encodes genetic information** — t(11;14) detection 88-92% AUROC
3. **Radiomics predicts outcome** — C-index 0.60-0.657, comparable to ISS/R-ISS
4. **Multimodal fusion outperforms single modalities** (morphology + flow + genomics)

### The Honest Admission
**Models trained on research cohorts fail when deployed to new hospitals** — 5-15% performance drop documented for foundation models across medical centers. No solution published.

### Most Important Unanswered Question
**Can computational pathology models be deployed clinically without retraining at each site?**

This is the critical gap between published accuracy (internal validation) and real-world utility.

---

## Research Gaps Identified

1. **Cross-center validation of t(11;14) prediction** — all published studies single-center
2. **Prospective randomized trial: radiomics-guided vs standard MM treatment** — clinical impact unproven
3. **Morphology+flow fusion for genetic prediction** — no published head-to-head comparison
4. **Foundation model robustness across medical centers** — domain shift unaddressed in literature
5. **MIL generalization to external MM cohorts** — transfer learning gap quantified but not solved

---

## For Pipeline Development

### Immediate Priorities (Based on Literature)
1. **Validate on external cohorts** — critical gap in current literature
2. **Address domain shift/stain variation** — major practical deployment blocker
3. **Multimodal integration** — morphology alone may be insufficient; combine with flow, radiomics
4. **Cell-level pretraining** — recent PCMMD dataset (5000+ cells) enables this
5. **Clinical decision support framework** — morphology as screening, not replacement for cytogenetics

### Recommended Public Datasets for Development
- **Primary:** CMB-MML (TCIA + dbGaP) — largest, most complete
- **Validation:** SN-AM + MM-Dataset (UFUA) for robustness testing
- **Benchmarking:** SegPC-2021 for cell-level tasks, PCMMD for morphologic features

### Foundation Models to Test
1. **UNI2-h** — best overall performance, good robustness
2. **Prov-GigaPath** — state-of-the-art but less robust to medical center differences
3. **CONCH** — vision-language alignment if reports available
4. **Virchow2** — most robust to domain shift (per 2025 study)

---

## Recommended Reading Order (For Newcomers)

**5-Minute Overview:**
→ literature_landscape.md, Section I (5-Minute Explainer)

**30-Minute Overview:**
→ literature_landscape.md, Sections F-J (Synthesis, Landscape, Knowledge Map)

**Detailed Understanding:**
→ literature_landscape.md, Sections A-E (Papers, clusters, contradictions, lineage, questions)

**Benchmark Reference:**
→ benchmark_targets.md (all key metrics at a glance)

**Dataset Selection:**
→ dataset_catalog.md, Sections 5-6 (selection guide, quick-start)

---

## Files Statistics

| Document | Lines | Size | Key Content |
|----------|-------|------|------------|
| literature_landscape.md | 847 | 57 KB | 40+ papers, 7 clusters, contradictions, synthesis, landscape |
| benchmark_targets.md | 380 | 22 KB | 100+ benchmark metrics with citations |
| dataset_catalog.md | 534 | 20 KB | 6 public datasets, access instructions, use-case guide |
| **TOTAL** | **2,279** | **99 KB** | **Publication-quality landscape mapping** |

---

## Contact & Version

**Created:** March 30, 2026
**Role:** PhD Researcher 1 — Imaging Pathology & Radiomics Surrogate-Genetics Pipeline
**Research Area:** Multiple Myeloma computational pathology and radiomics

**Questions/Updates:**
- For literature updates: Review TCIA repositories and new Nature Medicine publications quarterly
- For benchmark updates: Check arXiv preprints and conference proceedings (CVPR, NeurIPS, ICML)
- For dataset access: Visit TCIA.cancerimagingarchive.net and dbGaP portal directly

---
