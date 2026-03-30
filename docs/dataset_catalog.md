# Public Dataset Catalog: Multiple Myeloma Imaging & Pathology

## Overview

This document catalogs all publicly available datasets relevant to the MM imaging pathology and radiomics pipeline. Datasets are organized by modality with access instructions, licensing, and annotation details.

---

## 1. Histopathology (Microscopy) Datasets

### 1.1 TCIA CMB-MML: Cancer Moonshot Biobank - Multiple Myeloma Collection

**Basic Information**
- **Collection Name:** CMB-MML
- **Institution:** National Cancer Institute / The Cancer Imaging Archive (TCIA)
- **URL:** https://www.cancerimagingarchive.net/collection/cmb-mml/
- **dbGaP Link:** https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs002192.v1.p1

**Data Description**
- **Patient Count:** ≥1000 MM patients (NCI Cancer Moonshot Biobank initiative)
- **Imaging Modalities:** Radiology (PET/CT, MRI, CT) AND Histopathology
- **Histopathology Format:** Whole-slide images + DICOM-converted microscopy images
- **Associated Data:** Genomic, phenotypic, clinical data hosted on dbGaP

**Sample Details**
- Longitudinal collection during standard-of-care cancer treatment
- Multiple timepoints per patient (baseline, during treatment, post-treatment)
- Biospecimens: blood and tissue from medical procedures

**Data Annotations**
- Clinical staging information available
- Genomic annotations linked via dbGaP (FISH, cytogenetics, sequencing)
- Response assessment (disease burden, treatment response)

**Access Requirements**
- Registration on TCIA website
- For genomic/clinical data: dbGaP registration and approval required (may require IRB review)
- Terms: De-identified; publicly available after approval

**Licensing**
- Creative Commons Attribution 4.0 (CC-BY 4.0)
- Academic use permitted; publication requires acknowledgment

**File Format**
- DICOM format for radiology and whole-slide images
- Compatible with standard pathology software (QuPath, Aperio, etc.)

**Size Estimate**
- Radiology: Multiple terabytes (1000+ patients × multiple modalities × multiple timepoints)
- Histopathology: Large GB-scale dataset (exact size not specified on TCIA page)

---

### 1.2 MiMM_SBILab: Microscopic Images of Multiple Myeloma (TCIA)

**Basic Information**
- **Collection Name:** MiMM_SBILab
- **Institution:** The Cancer Imaging Archive (TCIA)
- **URL:** https://www.cancerimagingarchive.net/collection/mimm_sbilab/
- **Wiki:** https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756988
- **DOI:** 10.7937/tcia.2019.of2w8lxr (referenced in stain normalization datasets)

**Data Description**
- **Patient Count:** Multiple MM patients
- **Image Count:** 85 bone marrow aspirate microscopy images
- **Staining Protocol:** Jenner-Giemsa stain
- **Magnification:** 1000x
- **Image Size:** 2560×1920 pixels (raw BMP format)
- **Microscope:** Nikon Eclipse-200

**Sample Details**
- Bone marrow aspirate smears from confirmed MM patients
- Fresh preparations; standardized staining
- Variability sufficient for stain normalization testing (see stain normalization use case)

**Data Annotations**
- No pixel-level cell annotations provided
- Image-level diagnosis (MM confirmed)
- Intended use case: stain normalization benchmarking rather than cell detection

**Access Requirements**
- Open access; download directly from TCIA
- Registration on TCIA recommended

**Licensing**
- CC-BY 4.0 or equivalent TCIA open-access license

**File Format**
- Raw BMP format (2560×1920)
- Can be converted to standard formats (PNG, TIFF) for processing

**Size Estimate**
- ~400-500 MB total (85 images × 2560×1920 BMP files)

**Use Cases**
- Stain normalization algorithm development and benchmarking
- WSI preprocessing pipeline validation
- Foundation model robustness testing

---

### 1.3 SN-AM: Stain Normalization Bone Marrow Dataset (TCIA)

**Basic Information**
- **Collection Name:** SN-AM (White Blood Cancer - B-ALL and MM for Stain Normalization)
- **Institution:** The Cancer Imaging Archive (TCIA)
- **URL:** https://www.cancerimagingarchive.net/collection/sn-am/
- **Wiki:** https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52757009
- **DOI:** 10.7937/tcia.2019.of2w8lxr

**Data Description**
- **Patient Count:** Multiple B-ALL and MM patients
- **MM Image Count:** 100 images (confirmed)
- **B-ALL Image Count:** 90 images
- **Total Images:** 190 bone marrow aspirate images
- **Staining Protocol:** Jenner-Giemsa stain (same as MiMM_SBILab)
- **Magnification:** 1000x
- **Image Size:** 2560×1920 pixels (raw BMP format)
- **Microscope:** Nikon Eclipse-200 with digital camera

**Sample Details**
- Bone marrow aspirate smears from B-ALL and MM patients
- High staining variability across images (intentional design)
- Sufficient diversity for stain normalization methodology testing

**Data Annotations**
- **Additional Annotations:** Nucleus mask images provided (pixel-level)
- **Additional Annotations:** Background mask images provided
- **Nucleus Annotation:** Enables cell/nucleus-level analysis
- **Diagnosis:** Disease labels (B-ALL vs. MM)

**Access Requirements**
- Open access; direct download from TCIA
- Registration recommended

**Licensing**
- CC-BY 4.0

**File Format**
- Raw BMP (2560×1920 pixels)
- Accompanying mask files in same format

**Size Estimate**
- ~1 GB total (190 images × 2560×1920 + mask pairs)

**Use Cases**
- **Stain normalization algorithm development** (primary use case)
- Nucleus/cell segmentation benchmarking (masks provided)
- Background removal and preprocessing
- Foundation model robustness evaluation on stain-shifted images

**Research Impact**
- Widely cited in digital pathology preprocessing literature
- Standard benchmark for stain normalization method evaluation

---

### 1.4 PCMMD: Plasma Cell Morphology Dataset for Multiple Myeloma (Recent)

**Basic Information**
- **Collection Name:** PCMMD
- **Publication Venue:** Scientific Data / Nature (2025)
- **Title:** "A Novel Dataset of Plasma Cells to Support the Diagnosis of Multiple Myeloma"
- **Status:** Recently published (January 2025)

**Data Description**
- **Cell Count:** 5,000+ plasma and non-plasma cells (expert-labeled)
- **Image Source:** Bone marrow aspirate slides
- **Cell-Level Annotations:** Expert pathologist labels (plasma vs. non-plasma; morphologic features)
- **Patient Count:** Multiple MM patients + healthy controls

**Sample Details**
- Individual cell images extracted from aspirate smears
- Morphologic detail suitable for classification
- Diverse cell types (normal hematopoietic cells + clonal plasma cells)

**Data Annotations**
- **Expert Labels:** Plasma cell status
- **Patient Diagnostics:** Associated diagnostic information
- **Morphologic Features:** Potentially: nuclear size, maturity, cytoplasm characteristics

**Access Requirements**
- New dataset; likely open-access or CC-BY through Nature Scientific Data
- Link: https://www.nature.com/articles/s41597-025-04459-1

**Licensing**
- CC-BY 4.0 (standard for Nature Scientific Data)

**File Format**
- Not yet specified in abstract; likely PNG/TIFF or H5 format

**Size Estimate**
- ~500 MB - 2 GB (5000+ single-cell images)

**Use Cases**
- **Cell classification benchmarking** (plasma vs. non-plasma detection)
- **Morphologic feature extraction** for genetic prediction
- **MIL model training** on cell-level weak supervision
- **Foundation model fine-tuning** on hematology-specific tasks

**Significance**
- First large-scale, systematically-annotated plasma cell dataset
- Fills gap between generic histology datasets and MM-specific training needs

---

### 1.5 Multiple-Myeloma-Dataset (LabIA-UFBA/Hugging Face)

**Basic Information**
- **Collection Name:** Multiple-Myeloma-Dataset
- **Institution:** Laboratory of Intelligence in Algorithms (LabIA), Federal University of Bahia (UFBA)
- **GitHub:** https://github.com/LabIA-UFBA/Multiple-Myeloma-Dataset
- **Availability:** Hugging Face Hub (accessible and documented)

**Data Description**
- **Image Count:** Multiple images of bone marrow aspirate cells
- **Cell Types:** Plasma cells + other hematopoietic cells (multi-class)
- **Capture Method:** Smartphone microscopy + clinical microscope
- **Staining Protocol:** Wright-Giemsa stain (alternative to Jenner-Giemsa)

**Sample Details**
- Bone marrow aspirate smears from MM patients
- Captured using smartphone devices from clinical microscopes
- Realistic heterogeneity: variable lighting, focus, magnification

**Data Annotations**
- **Cell-Level Labels:** Individual cells labeled by type/category
- **Public Release:** Labeled dataset with class annotations

**Access Requirements**
- Public repository; direct GitHub/Hugging Face download
- No registration required

**Licensing**
- Specific license TBD; typically CC-BY or permissive open-source
- Academic use permitted

**File Format**
- Standard image formats (PNG, JPG)
- Cell-level crops or full-slide images (exact format TBD by repo structure)

**Size Estimate**
- Not precisely specified; likely 100 MB - 1 GB based on typical benchmarks

**Use Cases**
- **Cell detection and classification** on smartphone microscopy data
- **Domain adaptation testing** (smartphone vs. clinical microscope)
- **Low-resource deployment** scenarios
- **Practical ML application** for resource-limited settings

---

### 1.6 SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells Challenge Dataset

**Basic Information**
- **Collection Name:** SegPC-2021
- **Challenge Venue:** Medical imaging competition/challenge
- **Publication:** ScienceDirect (2022)
- **DOI/Reference:** Available in histopathology challenge databases

**Data Description**
- **Image Count:** 775 total images
  - 690 images of size 2040×1536 pixels
  - 85 images of size 1920×2560 pixels
- **Capture Setup:** Two different microscope + camera combinations
- **Staining:** Wright-Giemsa or Jenner-Giemsa (standard hematology stain)
- **Magnification:** Clinical microscopy (1000x or equivalent)

**Sample Details**
- Bone marrow aspirate images from MM patients
- Real-world variability from two different capture systems
- Designed to test robustness to equipment differences

**Data Annotations**
- **Segmentation Masks:** Pixel-level plasma cell annotations
- **Cell-Level Boundaries:** Nucleus and cytoplasm segmentation
- **Multiple Annotators:** Likely consensus or expert annotations

**Access Requirements**
- Challenge dataset; may require registration on challenge platform
- Post-challenge, typically released publicly or via institutional request

**Licensing**
- Typically CC-BY or academic-use-only through challenge terms

**File Format**
- Images + segmentation masks (likely PNG/TIFF or binary mask format)

**Size Estimate**
- ~200-500 MB (775 images × 2040×1536 or 1920×2560 + masks)

**Use Cases**
- **Cell segmentation benchmarking**
- **Instance segmentation** (distinguish individual plasma cells)
- **Robustness evaluation** across microscopy equipment
- **MIL or weakly-supervised learning** (cell boundaries from segmentation)

**Research Impact**
- Large benchmark challenge; identifies state-of-the-art segmentation methods
- Data heterogeneity (two systems) important for generalization testing

---

## 2. Radiomics Datasets (PET/CT)

### 2.1 CMB-MML (Radiology Component): PET/CT for MM

**Covered Under:** TCIA CMB-MML (Section 1.1)
- Radiology modality includes **PET/CT** scans from ≥1000 MM patients
- Full-body whole-body imaging for MM assessment
- Baseline and longitudinal scans

**Specific Details for Radiomics**
- **Modalities:** [18F]FDG PET + CT (same examination)
- **Number of Patients:** ≥1000 NDMM and treated MM
- **Clinical Data:** ISS, R-ISS, cytogenetics (FISH, sequencing)
- **Endpoints:** Overall survival, progression-free survival available via dbGaP

**Research Use Cases**
- PET/CT radiomics model development (as published in feasibility studies)
- Bone marrow segmentation benchmark
- Automated lesion detection validation

---

### 2.2 Public MM Radiomics Studies (Data-Derived Benchmarks)

**Note:** While some radiomics studies reference large cohorts (N=443 OS prediction study), raw imaging data is often not publicly released due to privacy/HIPAA restrictions. However, benchmark metrics and algorithms are published.

| Study | Cohort Size | Data Status | Accessibility |
|-------|------------|------------|---|
| Radiomics Feature Analysis for Survival (2025) | N=443 NDMM | Not publicly released | Metrics published; request to authors |
| Feasibility Study PET/CT Radiomics (2025) | N=129 NDMM | CMB-MML likely source | CMB-MML dataset |
| Cluster Analysis Autoencoder PET/CT (2023) | ~100-200 patients | Internal cohort | Not public |

**Recommendation:** CMB-MML is the primary public source; other radiomics studies may share data upon request to corresponding authors.

---

## 3. Multi-Modality Integrated Datasets

### 3.1 TCGA (The Cancer Genome Atlas) - Hematologic Malignancies

**Basic Information**
- **Project Name:** The Cancer Genome Atlas
- **Relevant Collection:** Acute Myeloid Leukemia (LAML), Lymphomas
- **URL:** https://portal.gdc.cancer.gov/
- **Note:** No dedicated MM project; closest is LAML for hematology validation

**Data Modalities**
- Histopathology slides (WSI in GDC portal)
- Whole-genome sequencing
- RNA-seq
- Clinical annotations

**Access Requirements**
- GDC Portal registration
- dbGaP approval for controlled-access data

**Relevant for MM Pipeline**
- Reference for MIL and foundation model benchmarking
- Hematology-specific task validation
- No MM-specific annotations but methodology transferable

---

### 3.2 CMB-MML Multimodal Integration

**Unique Advantage:** CMB-MML integrates:
1. **Radiology:** PET/CT, MRI, CT
2. **Histopathology:** Bone marrow aspirate/biopsy slides
3. **Genomics:** FISH, sequencing results (dbGaP)
4. **Clinical Data:** ISS, IMWG response assessment, survival
5. **Longitudinal Data:** Multiple timepoints

**Integration Potential**
- Morphology + radiomics fusion
- Genomics + imaging correlation
- Multimodal prognostic models

---

## 4. Summary Table: All Datasets at a Glance

| Dataset Name | Modality | Sample Count | Annotation Type | Access | Licensing |
|--------------|----------|--------------|-----------------|--------|-----------|
| **CMB-MML** | Radiology + Histopathology | 1000+ patients | Genomic, clinical, imaging | TCIA + dbGaP | CC-BY 4.0 |
| **MiMM_SBILab** | BM Aspirate Microscopy | 85 images | Diagnosis label | TCIA | CC-BY 4.0 |
| **SN-AM** | BM Aspirate Microscopy | 190 images (100 MM) | Nucleus/background masks | TCIA | CC-BY 4.0 |
| **PCMMD** | Cell-level (plasma cells) | 5000+ cells | Expert morphologic labels | Nature/public | CC-BY 4.0 |
| **MM-Dataset (UFBA)** | BM Aspirate Microscopy | Multiple | Cell type labels | GitHub/HuggingFace | Open |
| **SegPC-2021** | BM Aspirate Microscopy | 775 images | Segmentation masks | Challenge DB | Academic |
| **TCGA-LAML** | WSI + Genomics | 200+ AML patients | Genomic, clinical | GDC/dbGaP | CC-BY 4.0 |

---

## 5. Dataset Selection Guide for MM Pipeline

### For H&E Morphology-to-Genetic Prediction Tasks

**Recommended Primary Dataset:** CMB-MML (histopathology + genomic annotations)
- **Why:** Largest scale, confirmed MM diagnosis, genetic reference standard (FISH/sequencing)
- **Alternative:** SN-AM or MiMM_SBILab for preprocessing/stain variation study
- **Emerging:** PCMMD for cell-level morphologic feature extraction

### For WSI-Level Classification Tasks

**Recommended:** CMB-MML whole-slide images
- **Backup for stain robustness:** SN-AM (intentional stain heterogeneity)

### For Cell-Level Segmentation & Classification

**Recommended:** SegPC-2021 (775 images with masks)
- **Alternative:** PCMMD (5000+ cells, recent) for morphologic classification
- **Domain adaptation:** MM-Dataset (UFBA) for smartphone microscopy robustness

### For PET/CT Radiomics

**Only Public Source:** CMB-MML radiomics subset
- Access via TCIA (radiology) + dbGaP (outcomes, cytogenetics)
- Contact study team for specific radiomics features if published in papers

### For Multimodal Integration

**Best Resource:** CMB-MML
- Includes radiology (PET/CT), histopathology, genomics, clinical data
- Single-source coherence important for multimodal learning

### For Domain Adaptation & Robustness Testing

**Recommended Protocol:**
1. Train on **CMB-MML** (large, representative)
2. Validate on **SN-AM** (stain variation)
3. Test on **UFUA MM-Dataset** (equipment variation from smartphones)
4. Benchmark on **SegPC-2021** (multi-system capture)

---

## 6. Data Access & Download Instructions

### TCIA Datasets (CMB-MML, MiMM_SBILab, SN-AM)

1. Visit https://www.cancerimagingarchive.net/
2. Register free account
3. Search for collection name (CMB-MML, SN-AM, etc.)
4. Download directly or use NBIA Data Retriever for large datasets

### CMB-MML Genomic/Clinical Data (dbGaP)

1. Go to https://www.ncbi.nlm.nih.gov/gap/
2. Search study ID: phs002192.v1.p1
3. Apply for access (may require IRB approval for controlled data)
4. Download via authenticated access after approval

### GitHub Datasets (LabIA-UFBA MM-Dataset)

1. Visit https://github.com/LabIA-UFBA/Multiple-Myeloma-Dataset
2. Clone repository or download ZIP
3. Access also available on Hugging Face Hub

### PCMMD Dataset

1. Visit Nature Scientific Data article (2025)
2. Access supplementary data files
3. Register on repository site (TBD, likely figshare or Zenodo)

---

## 7. Important Notes & Licensing Compliance

### Data Privacy & Ethics
- All TCIA datasets are **de-identified** per HIPAA standards
- CMB-MML genomic data on dbGaP has controlled access; IRB approval may be required
- Appropriate attribution required per CC-BY 4.0 licenses

### Recommended Citation Format
- **CMB-MML:** Cite TCIA collection paper + dbGaP study ID
- **SN-AM:** "SN-AM Dataset: White Blood Cancer Dataset for Stain Normalization" + DOI: 10.7937/tcia.2019.of2w8lxr
- **MiMM_SBILab:** Cite TCIA collection
- **PCMMD:** Cite Nature Scientific Data article (2025)
- **UFBA MM-Dataset:** Cite GitHub repository + acknowledge LabIA-UFBA

### File Format Compatibility
- All microscopy images convertible to standard formats (PNG, TIFF, JPG)
- DICOM files (TCIA radiology) require specialized software (QuPath, Fiji, SimpleITK, etc.)
- Python libraries recommended: pydicom, tifffile, PIL, OpenCV

---

## 8. Limitations & Future Directions

### Current Gaps in Public Datasets
1. **No single large public MM dataset with complete multimodal data** (radiology + histopathology + genomics in one cohort)
   - CMB-MML closest but genomics behind dbGaP wall
2. **Limited MM-specific ground truth annotations** (most datasets are single modality)
3. **No cross-center standardized dataset** (robustness testing dataset lacking)
4. **Radiomics raw imaging data rarely public** (risk-benefit analysis by studies)

### Recommendations for Future Data Curation
1. **Create open MM data repository** with CMB-MML-level integration but simplified access
2. **Develop multi-center stain/equipment standardization dataset** (10+ institutions)
3. **Publish raw radiomics data** alongside feature extraction code
4. **Establish MM cell morphology ground truth** at scale (10,000+ annotated cells)
5. **Share foundation model embeddings** on published MM datasets for reproducibility

---

## Appendix: Quick-Start Guide

### Scenario 1: I Want to Train a t(11;14) Detector from Scratch
**Dataset Choice:** CMB-MML (histopathology) + SegPC-2021 (cell segmentation masks)
**Approach:**
1. Download CMB-MML WSI (TCIA)
2. Use SegPC-2021 for cell-level segmentation preprocessing
3. Extract patches, train MIL model, validate on held-out CMB-MML cohort

### Scenario 2: I Want to Test Stain Normalization Methods
**Dataset Choice:** SN-AM (intentional stain variation)
**Approach:**
1. Download SN-AM (100 MM + 90 B-ALL images)
2. Apply normalization methods (Macenko, Reinhard, etc.)
3. Evaluate visual quality and downstream classification accuracy

### Scenario 3: I Want to Build a Multimodal Morphology+Radiomics Model
**Dataset Choice:** CMB-MML (complete multimodal)
**Approach:**
1. Register on TCIA + dbGaP
2. Download histopathology slides (TCIA)
3. Download PET/CT images (TCIA)
4. Link to genomic/outcome data via dbGaP
5. Co-register imaging modalities; extract features; train multimodal model

---

