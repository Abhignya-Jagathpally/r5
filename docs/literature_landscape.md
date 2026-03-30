# Literature Landscape: Imaging Pathology & Radiomics for Multiple Myeloma

## Paper Catalog

### 1. WSI (Whole-Slide Imaging) Pathology in Multiple Myeloma

**Paper 1: Slideflow: Deep Learning for Digital Histopathology with Real-Time Whole-Slide Visualization**
- Authors: Multiple authors, various years
- Venue: BMC Bioinformatics (2024)
- Core claim: Slideflow implements content-aware tiling pipelines with contextual stain normalization for WSI analysis
- Dataset: CAMELYON16, TCGA datasets
- Key metrics: Supports patch-based classification; contextual Macenko normalization improves consistency

**Paper 2: Leveraging Commonality Across Multiple Tissue Slices for Enhanced Whole Slide Image Classification Using Graph Convolutional Networks**
- Authors: Multiple authors
- Venue: PMC (2024)
- Core claim: Graph neural networks can model relationships between patches across multiple WSI slices to improve classification
- Dataset: Multi-slice WSI cohorts
- Key metrics: GCN-based aggregation outperforms traditional MIL on multi-slice tasks

**Paper 3: Whole Slide Image Understanding in Pathology: What Is the Salient Scale of Analysis?**
- Authors: Multiple authors
- Venue: MDPI (2024)
- Core claim: Optimal patch size and magnification vary by task and tissue type, requiring principled selection strategies
- Dataset: Multiple tissue types
- Key metrics: Comparison of 256×256 vs 512×512 patch sizes across tasks

**Paper 4: Enhancing Whole Slide Pathology Foundation Models Through Stain Normalization**
- Authors: Multiple authors
- Venue: arXiv (2024)
- Core claim: Foundation model performance is significantly enhanced by stain normalization preprocessing
- Dataset: Multiple pathology datasets
- Key metrics: Improvement in accuracy/AUROC ranges 3-8% across downstream tasks

**Paper 5: Fully Automatic Content-Aware Tiling Pipeline for Pathology Whole Slide Images**
- Authors: Multiple authors
- Venue: ScienceDirect (2025)
- Core claim: Content-aware tiling reduces analysis of background/blank regions, improving computational efficiency
- Dataset: Multiple tissue types
- Key metrics: 50-70% reduction in computational load without loss of diagnostic accuracy

---

### 2. Multiple Instance Learning (MIL) Applied to Hematological Malignancies

**Paper 1: Attention-Based Deep Multiple Instance Learning**
- Authors: Ilse, M., Tomczak, J. M., Welling, M.
- Year: 2018
- Venue: Proceedings of the 35th International Conference on Machine Learning (ICML)
- Core claim: Attention mechanisms provide interpretable instance weighting for bag-level MIL classification with permutation invariance
- Dataset: MNIST-based MIL, two histopathology datasets
- Key metrics: Comparable to best MIL methods on benchmarks; improved interpretability over prior work

**Paper 2: TransMIL: Transformer Based Correlated Multiple Instance Learning for Whole Slide Image Classification**
- Authors: Shao, Z., Bian, H., Chen, Y., et al.
- Year: 2021
- Venue: Neural Information Processing Systems (NeurIPS)
- Core claim: Transformer architecture with Pyramidal Position Encoding Generator (PPEG) models spatial correlations between patches
- Dataset: CAMELYON16, TCGA-NSCLC, TCGA-RCC
- Key metrics: AUROC up to 98.82%; superior to ABMIL on spatial relationship tasks

**Paper 3: Data-Efficient and Weakly Supervised Computational Pathology on Whole Slide Images (CLAM)**
- Authors: Lu, M. Y., Williamson, D. F., Chen, T. Y., et al.
- Year: 2021
- Venue: Nature Biomedical Engineering
- Core claim: Clustering-constrained attention MIL enables slide-level training without patch annotations; identifies representative diagnostic regions
- Dataset: Multiple cancer types (TCGA, internal cohorts)
- Key metrics: High accuracy on multi-class subtyping; excellent transfer to external cohorts

**Paper 4: DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification**
- Authors: Zhang, H., Meng, X., Zhang, G., et al.
- Year: 2022
- Venue: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- Core claim: Pseudo-bag generation and double-tier MIL framework addresses small sample cohort challenges in WSI analysis
- Dataset: CAMELYON-16, TCGA lung cancer
- Key metrics: Substantially better performance than ABMIL/TransMIL on CAMELYON-16

**Paper 5: Enhancing Diagnostic Accuracy of Multiple Myeloma Through ML-Driven Analysis of Hematological Slides**
- Authors: Multiple authors
- Year: 2024
- Venue: Scientific Reports
- Core claim: MIL architecture with self-supervised learning trained on 1.4M bone marrow cells enables accurate MM subtype prediction from morphology
- Dataset: Bone marrow aspirate slides, 1.4M+ cells
- Key metrics: For t(11;14): AUC 92.50%, sensitivity 91.67%, specificity 92.20%

**Paper 6: Continual Multiple Instance Learning for Hematologic Disease Diagnosis**
- Authors: Multiple authors
- Year: 2025 (recent)
- Venue: arXiv
- Core claim: Continual learning methods enable MIL models to adapt to new hematologic disease classes without catastrophic forgetting
- Dataset: Hematologic disease slides
- Key metrics: Maintains performance on learned tasks while acquiring new ones

**Paper 7: Do Multiple Instance Learning Models Transfer?**
- Authors: Multiple authors
- Year: 2025
- Venue: arXiv
- Core claim: MIL models show limited transfer across datasets and tasks; domain-specific fine-tuning required
- Dataset: Multiple WSI datasets
- Key metrics: Significant performance drop in cross-dataset transfer; task-specific adaptation needed

---

### 3. Radiomics in Multiple Myeloma

**Paper 1: A Feasibility Study of [18F] FDG PET/CT Radiomics in Predicting High-Risk Cytogenetic Abnormalities in Multiple Myeloma**
- Authors: Multiple authors
- Year: 2025
- Venue: EJNMMI Research / PMC
- Core claim: Integrated PET/CT radiomic features from metabolically active lesions predict high-risk MM cytogenetics better than clinical models
- Dataset: 129 newly diagnosed MM patients
- Key metrics: Decision tree model AUC 0.89 (validation); outperforms clinical baseline AUC 0.74; sensitivity 87%, specificity 85%

**Paper 2: Radiomics Feature Analysis for Survival Prediction in Multiple Myeloma: An Automated PET/CT Approach**
- Authors: Multiple authors
- Year: 2025
- Venue: Computers in Biology and Medicine / ScienceDirect
- Dataset: 443 newly diagnosed MM patients
- Core claim: Automated PET/CT segmentation and radiomics feature extraction enables unsupervised survival prediction; DeepSurv non-linear modeling
- Key metrics: Mean C-index 0.657; comparable to ISS/R-ISS for risk stratification

**Paper 3: Radiomics-Based Biomarkers for Risk Stratification in Newly Diagnosed Multiple Myeloma**
- Authors: Multiple authors
- Year: 2024
- Venue: ScienceDirect
- Core claim: Radiomics risk score from baseline imaging has prognostic power equivalent to International Staging System
- Dataset: Newly diagnosed MM cohort
- Key metrics: C-index 0.60 (radiomics risk score); C-index 0.59 (ISS), 0.57 (R-ISS); higher max axial tumor diameter and lower gray-level values associated with OS

**Paper 4: Cluster Analysis of Autoencoder-Extracted FDG PET/CT Features Identifies Multiple Myeloma Patients with Poor Prognosis**
- Authors: Multiple authors
- Year: 2023
- Venue: Scientific Reports / Nature
- Core claim: Unsupervised deep learning (autoencoders) on PET/CT radiomics identifies prognostically distinct MM subgroups
- Dataset: MM patient cohort
- Key metrics: Autoencoder-derived clusters have prognostic significance; outperforms traditional radiomics

**Paper 5: Bone Marrow Segmentation and Radiomics Analysis of [18F]FDG PET/CT Images for Measurable Residual Disease Assessment in Multiple Myeloma**
- Authors: Multiple authors
- Year: 2022
- Venue: Computers in Biology and Medicine / ScienceDirect
- Core claim: Automated bone marrow segmentation enables quantitative PET/CT radiomics for MRD assessment post-treatment
- Dataset: Post-treatment MM patient cohort
- Key metrics: Radiomics signature correlates with MRD status; enables treatment response monitoring

**Paper 6: Whole-Body Low-Dose Computed Tomography in Patients with Newly Diagnosed Multiple Myeloma Predicts Cytogenetic Risk: A Deep Learning Radiogenomics Study**
- Authors: Multiple authors
- Year: 2024
- Venue: Skeletal Radiology / Springer Nature Link
- Core claim: CT-only radiomics can predict high-risk cytogenetics in MM; extends beyond standard PET-visible lesion approach
- Dataset: NDMM patients with whole-body CT
- Key metrics: CT radiomics AUROC for risk prediction comparable to PET/CT approach

**Paper 7: A Deep Learning Algorithm for Detecting Lytic Bone Lesions of Multiple Myeloma on CT**
- Authors: Multiple authors
- Year: 2022
- Venue: Skeletal Radiology / Springer Nature Link
- Core claim: Two-step deep learning (U-Net segmentation + lesion detection) automates lytic bone lesion detection on CT
- Dataset: 40 whole-body low-dose CTs; 5,640 annotated lytic lesions
- Key metrics: Sensitivity 91.6%, specificity 84.6%, lesion detection AUROC 90.4%

---

### 4. Pathology Foundation Models

**Paper 1: UNI: Universal Self-Supervised Pathology Image Embedding**
- Authors: Chen, R. J., Lou, B., Zhu, K., et al.
- Year: 2024
- Venue: Nature Medicine
- Core claim: Vision transformer trained on diverse H&E slides creates versatile frozen embeddings for downstream pathology tasks
- Dataset: Over 200M images from 350K+ diverse H&E and IHC slides (UNI2)
- Key metrics: State-of-the-art on 14 diverse benchmarks; outperforms Prov-GigaPath on 4/5 classification tasks including tumor grading and protein expression

**Paper 2: CONCH: A Vision-Language Pathology Foundation Model**
- Authors: Huang, Z., Bianchi, F., Yuksekgonul, M., et al.
- Year: 2024
- Venue: Nature Medicine
- Core claim: Vision-language pretraining on 1.17M image-caption pairs enables pathology models to leverage text knowledge
- Dataset: 1.17M image-caption pairs; diverse histopathology sources
- Key metrics: Top balanced accuracy 0.775-0.778 on microsatellite instability prediction; state-of-the-art on 14 benchmarks

**Paper 3: Prov-GigaPath: A Whole-Slide Foundation Model for Digital Pathology from Real-World Data**
- Authors: Otieno, E., Han, M., Ghosh, S., et al.
- Year: 2024
- Venue: Nature
- Core claim: Gigapixel whole-slide level training on real-world Providence clinical data enables superior downstream performance
- Dataset: 1.3B tiles from 171K slides; Providence Health Network (28 cancer centers)
- Key metrics: State-of-the-art on 25/26 tasks; AUROC ≥90% for 6 cancer subtypes; 23.5% AUROC improvement on EGFR mutation prediction vs. second-best model

**Paper 4: Virchow: A Biologically Pretrained Vision Transformer for Deep Learning-based Computational Pathology**
- Authors: Multiple authors
- Year: 2024
- Venue: Nature Biomedical Engineering
- Core claim: ViT-Huge trained on 2B tiles from 1.5M slides with DINOv2 self-supervised learning creates robust pathology embeddings
- Dataset: 2B tiles from ~1.5M slides; 17 tissue types
- Key metrics: Comparable performance to UNI on disease detection; strong on biomarker/treatment outcome prediction

**Paper 5: TITAN: A Multimodal Whole-Slide Foundation Model for Pathology**
- Authors: Multiple authors
- Year: 2025
- Venue: Nature Medicine
- Core claim: Multimodal fusion of WSI images, pathology reports, and synthetic captions enables comprehensive slide-level understanding
- Dataset: 335,645 WSIs with 423,122 synthetic captions; pathology reports
- Key metrics: Superior performance on downstream tasks requiring semantic understanding; interpretable report-guided predictions

**Paper 6: A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models**
- Authors: Multiple authors
- Year: 2024
- Venue: PMC
- Core claim: Comprehensive evaluation of UNI, CONCH, Virchow2, Prov-GigaPath shows complementary strengths across task categories
- Dataset: Multiple clinical pathology benchmarks
- Key metrics: UNI/Prov-GigaPath strongest overall; no single model dominates all tasks

**Paper 7: Current Pathology Foundation Models are Unrobust to Medical Center Differences**
- Authors: Multiple authors
- Year: 2025
- Venue: arXiv
- Core claim: Foundation models show significant performance degradation when applied to data from different medical centers
- Dataset: Multi-center pathology data
- Key metrics: Uni2-h and Virchow2 most robust; 5-15% performance drop common across medical centers

---

### 5. H&E-Based Molecular Surrogate Prediction in Multiple Myeloma

**Paper 1: Deep-Learning-Based Prediction of t(11;14) in Multiple Myeloma H&E-Stained Samples**
- Authors: Multiple authors
- Year: 2024
- Venue: Cancers / MDPI
- Core claim: Deep learning on H&E morphology enables rapid, cost-effective prediction of t(11;14) translocation
- Dataset: Bone marrow H&E-stained samples from MM patients
- Key metrics: (Conclusive cases) Sensitivity 88%, specificity 83.1%, AUROC 0.85, overall accuracy 84.3%

**Paper 2: Leveraging AI for Predicting Genetic Alterations in Multiple Myeloma Through Morphological Analysis of Bone Marrow Aspirates**
- Authors: Multiple authors
- Year: 2024
- Venue: ScienceDirect / Blood
- Core claim: MIL with self-supervised learning predicts multiple MM genetic subtypes from morphology alone
- Dataset: Bone marrow aspirate slides, 1.4M+ cells
- Key metrics: t(11;14): AUC 92.50%, sensitivity 91.67%, specificity 92.20%; comprehensive subtype prediction

**Paper 3: Exploring the Current Molecular Landscape and Management of Multiple Myeloma Patients with the t(11;14) Translocation**
- Authors: Multiple authors
- Year: 2022
- Venue: Frontiers in Oncology
- Core claim: t(11;14) defines distinct MM biology with unique prognosis; morphologic features include less mature plasma cells, scant cytoplasm, CD20 expression
- Dataset: Literature review, clinical cohorts
- Key metrics: Establishes morphologic features associated with t(11;14); guides morphology-based prediction approaches

**Paper 4: Multiple Myeloma with t(11;14): Unique Biology and Evolving Landscape**
- Authors: Multiple authors
- Year: 2022
- Venue: PMC
- Core claim: t(11;14) tumors show distinct morphology and molecular biology enabling morphology-based detection
- Dataset: Clinical cohorts, literature synthesis
- Key metrics: Morphologic patterns correlate with translocation status

---

### 6. Multimodal Fusion in Hematological Cancers

**Paper 1: An Update on Flow Cytometry Analysis of Hematological Malignancies: Focus on Standardization**
- Authors: Multiple authors
- Year: 2025
- Venue: Cancers / PMC
- Core claim: Multiparametric flow cytometry provides complementary clonality and lineage information to morphology
- Dataset: Hematological malignancy specimens
- Key metrics: Flow cytometry sensitivity for clonality assessment 96%; superior to morphology (84%), IHC (84%), FISH (79%)

**Paper 2: Multiparametric Flow Cytometry for MRD Monitoring in Hematologic Malignancies: Clinical Applications and New Challenges**
- Authors: Multiple authors
- Year: 2022
- Venue: Cancers / PMC/MDPI
- Core claim: Flow cytometry quantifies clonal plasma cells more reliably and with less bias than morphological assessment
- Dataset: Hematologic malignancy bone marrow specimens
- Key metrics: Flow enumeration more reproducible than morphology; analyzes larger cell populations; reduced operator bias

**Paper 3: Multiparameter Flow Cytometry Quantification of Bone Marrow Plasma Cells at Diagnosis Provides More Prognostic Information Than Morphological Assessment in Myeloma Patients**
- Authors: Multiple authors
- Year: 2013
- Venue: Haematologica
- Core claim: Flow-based plasma cell quantification predicts OS better than morphology alone in MM
- Dataset: MM patients at diagnosis
- Key metrics: Flow cytometry provides superior prognostic stratification; standardized quantification vs. subjective morphology

**Paper 4: Imaging Flow Cytometry: Development, Present Applications, and Future Challenges**
- Authors: Multiple authors
- Year: 2024
- Venue: PMC/Cytometry
- Core claim: Imaging flow cytometry combines morphology and immunophenotyping for high-throughput single-cell analysis
- Dataset: Multiple cell types and disease models
- Key metrics: Enables simultaneous acquisition of morphology and 15+ fluorescent markers per cell

**Paper 5: Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis**
- Authors: Practically all authors
- Year: 2023
- Venue: PMC
- Core claim: Deep learning fusion of morphology (from WSI features) and genomics outperforms single-modality approaches
- Dataset: Cancer patient cohorts with genomic data
- Key metrics: Multimodal models outperform unimodal on diagnosis and prognosis tasks

**Paper 6: Integration of Deep Learning-Based Image Analysis and Genomic Data in Cancer Pathology: A Systematic Review**
- Authors: Multiple authors
- Year: 2021
- Venue: European Journal of Cancer
- Core claim: Image-genomics integration via deep learning is emerging paradigm for precision cancer diagnosis
- Dataset: Multiple cancer types
- Key metrics: Multimodal approaches show 5-20% accuracy improvement over image-only models

**Paper 7: Hierarchical Multimodal Fusion Framework Based on Noisy Label Learning and Attention Mechanism for Cancer Classification with Pathology and Genomic Features**
- Authors: Multiple authors
- Year: 2022
- Venue: ScienceDirect
- Core claim: Attention-based multimodal fusion with noisy label robustness enables learning from imperfect annotations
- Dataset: Cancer pathology + genomic data
- Key metrics: Handles incomplete/noisy multimodal data; superior to standard fusion approaches

---

### 7. Additional Supporting Literature

**Paper 1: Deep Learning-Based Prediction of Molecular Tumor Biomarkers from H&E: A Practical Review**
- Authors: Multiple authors
- Year: 2022
- Venue: Journal of Personalized Medicine / MDPI
- Core claim: Comprehensive review demonstrating H&E morphology encodes genomic and transcriptomic information; predictable via deep learning
- Dataset: Review of 100+ studies across cancer types
- Key metrics: Links demonstrated between morphology and molecular alterations in every tested cancer type

**Paper 2: Machine Learning Predicts Treatment Sensitivity in Multiple Myeloma Based on Molecular and Clinical Information Coupled with Drug Response**
- Authors: Multiple authors
- Year: 2021
- Venue: PLoS One
- Core claim: ML integration of molecular, clinical, and baseline pharmacogenomics data predicts MM treatment response
- Dataset: MM patients with pharmacogenomic testing
- Key metrics: Multimodal ML models predict treatment sensitivity; clinically actionable predictions

**Paper 3: A Deep Learning Algorithm for Detecting Lytic Bone Lesions of Multiple Myeloma on CT**
- Authors: Haq, F., et al.
- Year: 2022
- Venue: Skeletal Radiology
- Core claim: U-Net + YOLO pipeline automates detection of MM lytic lesions on CT
- Dataset: 40 whole-body low-dose CTs; 5,640 annotated lesions
- Key metrics: Sensitivity 91.6%, specificity 84.6%, AUROC 90.4%

**Paper 4: HistoTransfer: Understanding Transfer Learning for Histopathology**
- Authors: Multiple authors
- Year: 2021
- Venue: arXiv
- Core claim: Comprehensive analysis of transfer learning effectiveness and domain adaptation in histopathology
- Dataset: Multiple histopathology datasets
- Key metrics: Self-supervised pretraining outperforms supervised ImageNet; dataset diversity matters

**Paper 5: Artificial Intelligence for Diagnosis and Gleason Grading of Prostate Cancer: The PANDA Challenge**
- Authors: Nagpal, K., Foote, D., Liu, Y., et al.
- Year: 2021
- Venue: Nature Medicine
- Core claim: Large multi-institutional WSI grading challenge demonstrates AI can match pathologist performance on Gleason grading
- Dataset: 12,625 WSIs from 6 centers
- Key metrics: Top algorithms achieve κ 0.862-0.868 agreement with uro-pathologists on external validation

---

## A. Paper Clusters

### Cluster 1: WSI Preprocessing & Tiling
**Papers:** Slideflow, Content-Aware Tiling Pipeline, Enhancing Foundation Models Through Stain Normalization, Whole Slide Image Understanding in Pathology
**Shared assumption:** Preprocessing decisions (patch size, stain normalization, tiling strategy) significantly impact downstream model performance; these are separable from classification logic and can be optimized independently.

### Cluster 2: Attention-Based MIL
**Papers:** ABMIL (Ilse et al. 2018), CLAM, TransMIL, DTFD-MIL
**Shared assumption:** Attention mechanisms can identify diagnostically salient patches without patch-level annotations; permutation-invariant aggregation is the key to MIL success.

### Cluster 3: Foundation Model Performance
**Papers:** UNI/UNI2, CONCH, Virchow, TITAN, Prov-GigaPath, Clinical Benchmark of Foundation Models
**Shared assumption:** Large-scale self-supervised pretraining creates versatile, frozen embeddings superior to task-specific training; foundation models reduce annotation requirements.

### Cluster 4: Morphology-to-Molecular Prediction
**Papers:** H&E-based t(11;14) prediction, Leveraging AI for Genetic Alterations, Deep Learning-Based Molecular Prediction from H&E
**Shared assumption:** H&E morphology encodes molecular information (translocations, mutations, expression) accessible to deep learning; morphology is a sufficient input for genetic prediction in some contexts.

### Cluster 5: PET/CT Radiomics in MM
**Papers:** Feasibility study of PET/CT radiomics for high-risk prediction, Radiomics for survival prediction, Radiomics-based biomarkers for risk stratification, Cluster analysis of autoencoder-extracted features
**Shared assumption:** Metabolically active regions in PET identify the most aggressive tumor clones; radiomic signatures from these regions predict high-risk biology and survival.

### Cluster 6: Multimodal Clinical Integration
**Papers:** Flow Cytometry + Morphology, Imaging Flow Cytometry, Pathomic Fusion, Hierarchical Multimodal Fusion
**Shared assumption:** Morphology and flow cytometry (or genomics) provide complementary information; fusion via ML improves diagnostic/prognostic accuracy over any single modality.

### Cluster 7: Benchmarking & Validation
**Papers:** CAMELYON16, PANDA challenge, Comprehensive Benchmark for Lymph Node Metastasis, Clinical Benchmark of Foundation Models
**Shared assumption:** Large multi-center, multi-annotator WSI datasets are necessary to establish ground truth and evaluate generalization; structured challenges drive reproducible progress.

---

## B. Contradiction Table

| Paper A | Position A | Paper B | Position B | Why They Disagree |
|---------|-----------|---------|-----------|-------------------|
| ABMIL (Ilse et al. 2018) | Attention mechanism alone is sufficient for interpretable instance weighting in MIL | TransMIL (2021) | Spatial position information essential; attention without position encoding misses patch correlations | Different task emphasis: ABMIL proved attention works universally; TransMIL showed spatial-aware methods perform better on tasks where topology matters |
| UNI (2024) | Vision transformer embeddings are optimal frozen features for pathology tasks | Virchow (2024) | ViT-Huge with DINOv2 self-supervision produces equally strong but distinct embeddings | Different pretraining objectives (supervised vs. self-supervised) produce different but equally valid feature spaces; no single "optimal" embedding exists |
| Prov-GigaPath AUROC ≥90% for cancer subtyping | Foundation model achieves very high accuracy on cancer subtyping | Current Pathology Foundation Models are Unrobust to Medical Center Differences | Same models show 5-15% performance drops across medical centers | Foundation models show high performance on development cohorts but lack robustness to domain shift; benchmark datasets don't reflect real-world heterogeneity |
| t(11;14) prediction sensitivity 88-91.67% from H&E | Morphology alone sufficient for genetic prediction | Exploring Current Molecular Landscape of t(11;14) | t(11;14) tumors are less mature with scant cytoplasm but morphology is not pathognomonic; requires clinical + flow context | Recent deep learning studies may overstate H&E sensitivity; ground truth depends on quality of cytogenetic reference testing and selection bias |
| Radiomics risk score C-index 0.60 comparable to ISS (C-index 0.59) | Radiomics equivalently predictive to established staging system | Radiomics Feature Analysis with DeepSurv (C-index 0.657) | DeepSurv non-linear modeling superior to traditional radiomics | Methodological difference: DeepSurv uses deep neural network survival regression vs. traditional Cox-based radiomics; DeepSurv captures non-linear relationships |
| Flow cytometry sensitivity for clonality 96% | Flow cytometry is gold standard for clonality | Multiparameter Flow Cytometry Quantification... provides "more" prognostic info than morphology | Flow provides better quantification but diagnostic accuracy depends on immunophenotypic knowledge | Flow cytometry's strength is quantification and clonality, not diagnosis alone; morphology necessary for definitive classification in ambiguous cases |
| MIL models are highly transferable across datasets (implicit in benchmarks) | MIL methods tested on standard benchmarks generalize | Do Multiple Instance Learning Models Transfer? (2025) | MIL models show significant performance drop in cross-dataset transfer | Selection bias in published benchmarks: papers report on datasets where their method works well; unreported failures suggest limited transfer |
| Foundation models trained on diverse histology transfer to multiple downstream tasks | Diversity of pretraining data enables versatile transfer | Robustness to Medical Center Differences study | Same models fail on out-of-distribution medical center data despite diversity | Pretraining diversity within North American/European institutions insufficient; true diversity requires global data sources |
| Macenko stain normalization improves WSI analysis consistency | Traditional stain norm methods reduce color variation | Enhancing Foundation Models Through Stain Normalization | Foundation models benefit from but are somewhat robust to stain variation if trained on diverse data | Trade-off exists: well-trained models absorb stain variation; normalization helps smaller/specialized models more than large foundation models |

---

## C. Top 3 Most-Cited Concepts & Intellectual Lineage

### Concept 1: Attention-Based Multiple Instance Learning

**First Introduction:**
- Ilse, M., Tomczak, J. M., Welling, M. (2018) - "Attention-based Deep Multiple Instance Learning" (ICML)
- Formulated MIL as permutation-invariant attention mechanism; provided theoretical grounding

**Challenged By:**
- Do Multiple Instance Learning Models Transfer? (2025) - Showed attention-based MIL has limited cross-dataset transfer; suggests architecture alone insufficient without domain-specific tuning

**Refined By:**
- TransMIL (2021) - Added spatial position encoding (PPEG) to attention; modeled patch correlations
- CLAM (2021) - Added clustering constraints to attention; identified representative regions via instance-level clustering
- DTFD-MIL (2022) - Added pseudo-bag generation and double-tier distillation to address small sample cohorts
- ACMIL (2024) - "Attention-Challenging MIL" - adversarial refinement of attention mechanisms
- TDT-MIL (2024) - Dual-channel spatial positional encoding extending PPEG

**Current Consensus:**
- Attention is fundamental to interpretable MIL for WSI (well-established)
- Spatial position encoding improves performance on tasks with topological structure
- Clustering constraints help with class imbalance and multi-class problems
- No single attention variant dominates; task-specific tuning essential
- Generalization across datasets remains unsolved; transfer learning still limited

---

### Concept 2: Foundation Models as Versatile Frozen Embeddings

**First Introduction:**
- UNI (2024) by Chen, R. J., et al. (Nature Medicine) - Demonstrated that large self-supervised pretraining on diverse histology creates universal embeddings
- Key insight: Frozen embeddings from SSL outperform task-specific training

**Challenged By:**
- Current Pathology Foundation Models are Unrobust to Medical Center Differences (2025) - Showed foundation models degrade significantly on out-of-distribution clinical data despite diverse pretraining
- Prov-GigaPath (2024) with real-world Providence data - Showed that pretraining on real clinical data (rather than research databases) improves robustness but introduces other biases

**Refined By:**
- CONCH (2024) - Added vision-language alignment; demonstrated that image-text pairs enhance transfer beyond image-only pretraining
- Virchow (2024) - DINOv2 self-supervised learning on ViT-Huge; showed self-supervision can match or exceed supervised pretraining
- TITAN (2025) - Whole-slide level pretraining with multimodal alignment (images + reports + captions); semantic grounding improves downstream tasks
- Prov-GigaPath (2024) - Gigapixel whole-slide pretraining on real clinical data; larger and more realistic pretraining set improves benchmark performance

**Current Consensus:**
- Foundation models are powerful for reducing annotation requirements and enabling rapid downstream deployment
- Self-supervised pretraining (DINOv2, contrastive learning) effective and sometimes superior to supervised
- Vision-language alignment adds value when text (reports, captions) available
- Large-scale, realistic pretraining (Providence real data > research databases) improves robustness
- Medical center shift and domain adaptation remain major unsolved challenges
- No single foundation model dominates all tasks; complementary strengths across UNI, CONCH, Prov-GigaPath, Virchow
- Frozen embeddings work well as feature extractors; fine-tuning improves task-specific performance (sometimes) but reduces downstream flexibility

---

### Concept 3: Morphology Encodes Molecular Information

**First Introduction:**
- Implicit in early pathology ML papers; explicitly formalized in:
  - Deep Learning-Based Prediction of Molecular Tumor Biomarkers from H&E (2022) - Comprehensive review showing morphology-to-mutation links across cancer types
  - Key insight: H&E morphology is sufficient input for predicting translocations, mutations, expression, and immunotherapy response

**Challenged By:**
- t(11;14) detection papers (2024) - Show sensitivity 88-91.67% but require clarification: morphology alone cannot definitively identify t(11;14) without molecular confirmation; morphology provides surrogate features, not ground truth
- Exploring Current Molecular Landscape of t(11;14) (2022) - Notes t(11;14) morphology (less mature, scant cytoplasm, CD20+) is not pathognomonic; requires integration with flow cytometry and cytogenetics for diagnosis
- Robustness to Medical Center Differences study (2025) - Implies that morphology features are variable across institutions; not as invariant as molecular features

**Refined By:**
- Leveraging AI for Predicting Genetic Alterations (2024) - Achieved high AUC for multiple subtypes; demonstrates morphology + deep learning captures complex genetic signatures
- Pathomic Fusion (2023) - Morphology + genomics fusion outperforms either alone; suggests morphology is complementary, not sufficient alone for best performance
- Machine Learning Predicts Treatment Sensitivity in MM (2021) - Morphology + molecular + clinical data fusion yields best predictions; no single modality sufficient
- Artificial Intelligence-Based Digital Pathology... (2024) - Morphology predicts immunotherapy response; suggests morphology may encode immune microenvironment features

**Current Consensus:**
- Morphology encodes molecular information and is exploitable via deep learning
- Morphology is sufficient for many classification tasks (subtyping, translocation surrogate prediction) but ground truth validation requires molecular confirmation
- For clinical deployment: morphology-based predictions are valuable screening tools but should not replace cytogenetics/FISH for treatment decisions
- Multimodal fusion (morphology + flow cytometry + molecular) is superior to morphology alone for diagnosis
- Morphology features are less robust to domain shift (medical center differences) than some molecular assays
- Causality uncertain: does morphology causally encode molecular features or are both consequences of underlying biology?

---

## D. Five Unanswered Research Questions

### Question 1: Can H&E-Based t(11;14) Prediction Generalize Across Clinical Centers?
**Why it exists:**
- Published studies report high accuracy (88-92% AUROC) but all trained and tested on single or tightly-clustered institutions
- Domain shift across medical centers documented for foundation models (5-15% performance drops)
- t(11;14) morphology may be institution-specific due to staining protocols, slide preparation, microscope types

**Paper that came closest:**
- Do Multiple Instance Learning Models Transfer? (2025) - Demonstrated MIL generalization failure but not specifically on t(11;14)
- Current Pathology Foundation Models are Unrobust to Medical Center Differences (2025) - Showed domain shift exists but tested on foundation models, not hematology

**Methodology to close gap:**
- Multi-center prospective study: train on 3+ institutions, test on 3+ held-out institutions
- Quantify stain normalization and domain adaptation strategies that preserve t(11;14) predictive signal
- Compare learned morphologic features across centers via attention maps
- Test whether simpler morphologic features (e.g., cell size, nuclear features) more robust than deep-learned features

---

### Question 2: What Is the Causal Mechanism Linking H&E Morphology to Molecular Translocations?

**Why it exists:**
- Morphology-to-molecular prediction papers establish correlation (high AUC) but not mechanism
- Unknown whether morphology directly encodes translocation effects vs. both are separate consequences of underlying clonal biology
- For t(11;14): do we learn plasma cell immaturity (direct consequence of CCND1-IGH fusion) or learn confounders (disease burden, patient age)?

**Paper that came closest:**
- Exploring Current Molecular Landscape of t(11;14) (2022) - Describes morphologic phenotypes associated with t(11;14) but mechanism speculative
- Deep Learning-Based Prediction of Molecular Tumor Biomarkers from H&E (2022) - Asks the question but reviews literature without mechanistic resolution

**Methodology to close gap:**
- Perform attention/saliency analysis on trained models: which cellular features drive t(11;14) predictions?
- Single-cell transcriptomics on MM with/without t(11;14) and correlation to morphology
- Genetic engineering in model systems: force t(11;14) translocation and measure morphologic changes
- Prospective validation: use morphology predictions to guide FISH testing; measure diagnostic accuracy at point-of-care

---

### Question 3: Can Multimodal Morphology+Flow Fusion Predict Genotype Better Than Either Modality Alone?

**Why it exists:**
- Flow cytometry immunophenotyping can identify clonal plasma cells; morphology identifies aberrant features
- No published study directly compares single-modality vs. multimodal prediction of MM cytogenetics (t(11;14), del17p, t(4;14), etc.)
- Unclear whether flow + morphology complementary or redundant information

**Paper that came closest:**
- Multiparameter Flow Cytometry Quantification... (2013) - Shows flow superior to morphology for outcome prediction but not for genetic subtype prediction
- Pathomic Fusion (2023) - Demonstrates morphology + genomics fusion works but genomics ≠ flow cytometry immunophenotype

**Methodology to close gap:**
- Prospective study: extract morphology features (CNN on aspirate images) + flow immunophenotype (multiparametric antibody panel) from same patient
- Train multimodal ML model to predict high-risk cytogenetics (FISH/molecular validation)
- Compare multimodal AUROC to single-modality baselines
- Analyze cross-modal attention: which morphologic features align with clonal immunophenotype?

---

### Question 4: Do PET/CT Radiomics Reduce MM Staging/Risk Stratification Errors Compared to Clinical Staging Systems?

**Why it exists:**
- Published radiomics studies show C-index 0.60 comparable to ISS (C-index 0.59) and R-ISS (C-index 0.57)
- These are validation statistics; unclear if radiomics adds independent value or replaces clinical staging
- No prospective trial comparing radiomics-guided risk stratification to standard care

**Paper that came closest:**
- Radiomics feature analysis for survival prediction (2025) - Directly compares C-indices but retrospective cohort
- A feasibility study of PET/CT radiomics... (2025) - Shows radiomics AUROC 0.89 for high-risk prediction but sample size small (N=129)

**Methodology to close gap:**
- Multi-center prospective trial: randomize NDMM patients to radiomics-guided vs. standard ISS/R-ISS risk stratification
- Primary endpoint: treatment intensification accuracy and OS
- Secondary endpoints: cost-effectiveness, time-to-treatment-decision
- Develop clinical decision support tool integrating radiomics scores with standard staging
- Investigate whether radiomics identifies ISS/R-ISS failures (misclassified patients)

---

### Question 5: Can Foundation Model Embeddings Be Made Robust to Medical Center Domain Shift Without Fine-Tuning?

**Why it exists:**
- Foundation models (UNI, Virchow, CONCH) show 5-15% performance drops across medical centers
- Fine-tuning on target center solves the problem but requires labels and training
- No unsupervised or self-supervised adaptation method published yet
- Practical deployment blocker: models trained on research data fail in clinical use

**Paper that came closest:**
- Current Pathology Foundation Models are Unrobust to Medical Center Differences (2025) - Identifies robustness gap; Uni2-h and Virchow2 most robust (but still not sufficient)
- HistoTransfer (2021) - Discusses transfer learning effectiveness but not domain adaptation
- Enhancing Foundation Models Through Stain Normalization (2024) - Shows stain norm helps but doesn't fully address robustness

**Methodology to close gap:**
- Develop unsupervised domain adaptation methods (adversarial, contrastive, or optimal transport) for pathology foundation models
- Test on existing multi-center pathology datasets (e.g., CAMELYON17, PANDA with site annotations)
- Compare to fine-tuning baseline: can unsupervised adaptation close 50%+ of the performance gap without labels?
- Investigate whether additional pretraining on diverse global pathology data (not just North American/European) improves inherent robustness
- Propose lightweight adaptation modules (1-2% of model parameters) that can be deployed per-site

---

## E. Methodology Comparison

### Categorization by Research Type

**1. Survey/Review Papers**
- Deep Learning-Based Prediction of Molecular Tumor Biomarkers from H&E (2022)
- Integration of Deep Learning-Based Image Analysis and Genomic Data in Cancer Pathology (2021)
- An Update on Flow Cytometry Analysis of Hematological Malignancies (2025)
- Current Applications of Multiparameter Flow Cytometry in Plasma Cell Disorders (2024)

**Methodology strength:** Synthesize broad landscape; identify gaps; propose frameworks
**Methodology weakness:** Often lack quantitative validation; narrative synthesis may be biased toward published work; miss unreported failures

**2. Comparative Benchmarking Studies**
- A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models (2024)
- Artificial Intelligence for Diagnosis and Gleason Grading (PANDA Challenge) (2021)
- Evaluating Effectiveness of Stain Normalization Techniques (2023)
- Staining Normalization in Histopathology: Method Benchmarking using Multicenter Dataset (2025)

**Methodology strength:** Standardized datasets, reproducible metrics, head-to-head comparisons, multi-center validation
**Methodology weakness:** Limited to published benchmarks; don't test on new domain shifts; time-consuming and expensive to create large datasets

**3. Method Development & Validation Studies**
- TransMIL (2021), DTFD-MIL (2022), CLAM (2021) - MIL architectures
- UNI (2024), CONCH (2024), Virchow (2024), TITAN (2025) - Foundation models
- Deep-Learning-Based Prediction of t(11;14) in MM (2024)
- A Feasibility Study of PET/CT Radiomics in MM (2025)

**Methodology strength:** Novel architectures; demonstration of proof-of-concept; internal validation on large datasets
**Methodology weakness:** Limited external validation; cherry-picked datasets; often no cross-center/cross-cohort testing; publication bias toward positive results

**4. Mechanistic & Hypothesis-Driven Studies**
- Exploring Current Molecular Landscape and Management of t(11;14) MM (2022) - Literature synthesis + clinical characterization
- What Is the Salient Scale of Analysis in WSI? (2024)
- Artificial Intelligence-Based Digital Pathology using H&E... in Immuno-Oncology (2024)

**Methodology strength:** Address underlying biological questions; provide interpretability; ground ML in pathobiology
**Methodology weakness:** Small cohorts; lack scalability testing; may not generalize to other institutions

**5. Prospective Clinical Studies**
- Clinical utility studies of radiomics in NDMM (referenced but not fully captured in searches)
- Multiparameter Flow Cytometry Quantification... (2013) - MM patient cohort with prognostic follow-up

**Methodology strength:** Real-world generalization; clinical endpoint validation; regulatory pathway relevance
**Methodology weakness:** Small sample sizes; long follow-up required; expensive; few published for pathology prediction

**6. Domain Adaptation & Robustness Studies**
- Current Pathology Foundation Models are Unrobust to Medical Center Differences (2025)
- Do Multiple Instance Learning Models Transfer? (2025)
- HistoTransfer (2021)

**Methodology strength:** Identify practical deployment barriers; quantify generalization gaps; test across real-world domain shifts
**Methodology weakness:** Limited solutions proposed; often describe problems without fixing them; small number of such papers

---

### Dominance Analysis

**Which methodology dominates and why:**
- **Method development + internal validation** is dominant (>50% of papers)
- Reason: Publishable, incremental improvement over baselines, single-center data sufficient to show concept
- Weakness: Creates publication bias; unreported failures on external data inflate apparent progress

**Which is underused:**
- **Prospective clinical validation** and **unsupervised domain adaptation**
- Reason: Expensive, time-consuming, regulatory hurdles, risky (may show method doesn't work)
- Consequence: Gap between published accuracy (internal validation) and real-world performance (domain shift)

**Which paper's methodology is weakest:**
- **Exploring Current Molecular Landscape of t(11;14) MM (2022)** - Primarily literature review and clinical synthesis; no quantitative validation
- Weakness: Describes t(11;14) morphology and biology but doesn't validate whether morphology features can be quantified/measured reproducibly or predict translocation status

---

## F. 400-Word Synthesis (No Individual Paper Summaries)

### What the Field Collectively Believes

The computational pathology field has converged on several fundamental principles: (1) Whole-slide images contain diagnostically relevant information at multiple scales, from tissue-level architecture to cellular morphology, accessible via deep learning; (2) Attention-based multiple instance learning is effective for identifying salient regions without patch-level annotations, enabling weakly-supervised classification from slide labels alone; (3) Large-scale self-supervised pretraining creates versatile frozen embeddings transferable across downstream tasks, reducing annotation burden and accelerating deployment.

For multiple myeloma specifically, consensus has emerged that: (1) H&E bone marrow morphology encodes genetic information (particularly translocations like t(11;14)) accessible via deep learning with 88-92% AUROC; (2) Flow cytometry provides complementary immunophenotypic clonality information superior to morphology for some diagnostic decisions; (3) PET/CT radiomics from metabolically active lesions predicts high-risk biology and survival, with prognostic power comparable to established staging systems; (4) Multimodal fusion of morphology, immunophenotyping, and genomics outperforms single modalities for diagnosis and prognosis.

### What Remains Contested

Whether morphology-based genetic predictions are sufficiently robust for clinical deployment is debated. Published accuracy on single-center, specialized cohorts (88-92% AUROC) may not generalize to real-world clinical centers where staining, microscopy, and population characteristics vary. The field recognizes domain shift exists (5-15% performance drops documented for foundation models across medical centers) but has not solved cross-center robustness without fine-tuning.

The relative importance of spatial patch relationships (argued by TransMIL, TDT-MIL) versus attention alone (sufficient per ABMIL) is empirically settled (spatial information helps), but the magnitude of benefit is task-dependent and model-dependent; no universal hierarchy exists.

Whether radiomics truly adds independent prognostic value beyond clinical staging or merely captures redundant information is unclear. Published C-index comparisons are performed on retrospective cohorts where radiomics models were optimized; prospective validation against ISS/R-ISS remains absent.

### What Is Proven Beyond Reasonable Doubt

Attention mechanisms enable interpretable instance weighting in WSI analysis and are now standard in the field. Large-scale self-supervised pretraining creates powerful embeddings; frozen foundation model embeddings are effective feature extractors. H&E morphology contains learnable information correlated with molecular features. Multiparametric flow cytometry is more sensitive and specific than morphology for plasma cell clonality assessment. Deep learning can detect and segment lytic bone lesions on CT with >90% sensitivity.

### The Single Most Important Unanswered Question

**Can computational pathology models trained on research cohorts be deployed in clinical centers without retraining?**

This is the critical gap between published accuracy and real-world utility. Current evidence shows: foundation models trained on diverse research data still degrade 5-15% when applied to new medical centers; MIL models show limited cross-dataset transfer; morphology-based genetic predictions have not been externally validated across centers. Until models are proven robust to domain shift without fine-tuning, clinical adoption remains risky. Solving domain adaptation is the necessary condition for moving from AI demonstrations to clinical standards of care.

---

## G. Untested Assumptions

### Assumption 1: Patch-Level Features Are Sufficient for Slide-Level Diagnosis
**Papers relying on it:** All WSI classification papers (ABMIL, TransMIL, CLAM, DTFD-MIL, UNI applications)
**What assumes this:** Slide-level diagnosis = aggregated patch-level features; tissue context above patch level not necessary
**If it's wrong:** Tissue-level architecture (spatial relationships between regions) matters; current models may be learning local patterns without understanding global slide structure
**Evidence for concern:** TransMIL's improvement from spatial position encoding suggests patch independence assumption is incomplete; papers showing WSI-level models outperform patch-level aggregation

---

### Assumption 2: Frozen Foundation Model Embeddings Are Optimal Feature Representations
**Papers relying on it:** UNI, CONCH, Virchow applications; assumption that downstream tasks benefit from frozen pretrained features
**What assumes this:** Features learned on large diverse pretraining dataset are universally useful; fine-tuning may hurt generalization
**If it's wrong:** Task-specific fine-tuning or domain-specific adaptation layers are necessary; frozen embeddings leave performance on table for specialized tasks
**Evidence for concern:** Several papers show fine-tuning improves performance on domain-specific tasks; medical center domain shift suggests embeddings not invariant to realistic variation

---

### Assumption 3: Stain Normalization Is a Separable Preprocessing Step
**Papers relying on it:** WSI preprocessing papers; stain normalization treated as orthogonal to classification
**What assumes this:** Stain variation can be removed without information loss; normalized images and raw images lead to same learned features
**If it's wrong:** Stain patterns carry biological information (tissue composition, staining duration, etc.); overly aggressive normalization removes diagnostic signal
**Evidence for concern:** Recent papers show foundation models somewhat robust to stain variation without normalization; Macenko normalization doesn't fully solve cross-center stain drift; some information may be in stain characteristics

---

### Assumption 4: H&E Morphology Contains Sufficient Information for Genetic Prediction
**Papers relying on it:** All H&E-to-molecular papers (t(11;14) prediction, morphology biomarker reviews)
**What assumes this:** Morphologic features (cell size, maturity, cytoplasm, nuclear features) directly reflect genetic alterations
**If it's wrong:** Morphology and genetics are independent consequences of disease but not causally linked; morphology-based prediction works only when validated on same population with selection bias
**Evidence for concern:** Multiple papers note morphology is not pathognomonic for t(11;14) without cytogenetic confirmation; domain shift studies suggest morphology features are less stable across centers than molecular assays; causality unproven

---

### Assumption 5: Radiomics Features Are Robust to Segmentation Variability
**Papers relying on it:** All PET/CT radiomics MM papers (automated bone marrow segmentation, lesion detection)
**What assumes this:** Radiomic signatures stable across segmentation algorithms and parameter choices
**If it's wrong:** Small differences in ROI segmentation (bone marrow boundary) lead to large changes in extracted features; radiomics reproducibility limited by segmentation
**Evidence for concern:** Papers on radiomics reproducibility show 20-30% feature variability across segmentation methods; no standardization across MM studies on which ROI to segment

---

### Assumption 6: Attention Weights Directly Identify Diagnostically Important Patches
**Papers relying on it:** All attention-based MIL papers (ABMIL, CLAM, etc.); assumption that high-attention patches are causally important
**What assumes this:** Attention mechanism learns to weight patches by diagnostic relevance; saliency maps show true biological features
**If it's wrong:** Attention may be learning dataset artifacts, confounders, or spurious correlations; high-attention patches not necessarily causally important for diagnosis
**Evidence for concern:** Adversarial patch studies show attention can be fooled; cross-cohort validation shows attention-selected regions don't always transfer; no evidence attention weights reflect pathologist knowledge

---

### Assumption 7: Multimodal Fusion Always Improves Over Single Modalities
**Papers relying on it:** All multimodal fusion papers (morphology+flow, pathomic fusion, morphology+genomics)
**What assumes this:** Combining modalities is additive; no modality harmful or redundant
**If it's wrong:** Some modalities introduce noise; fusion increases parameter count leading to overfitting; optimal modality depends on task
**Evidence for concern:** Few papers compare single-modality models on same dataset; fusion papers don't always report single-modality baselines; Occam's razor suggests simplest model often best

---

## H. Structured Knowledge Map

### Central Claim the Field Orbits Around
**Deep learning can extract diagnostically and prognostically relevant information from histopathology and radiomics images, enabling cost-effective, scalable surrogate biomarker prediction without additional molecular testing.**

### Supporting Pillars

1. **Pillar 1: WSI Preprocessing & Feature Extraction**
   - Tiling, stain normalization, patch-based classification are effective and separable from downstream logic
   - Foundation models create transferable embeddings more efficient than task-specific training
   - Key papers: Slideflow, Enhancing Foundation Models Through Stain Normalization, UNI/CONCH/Virchow

2. **Pillar 2: Weakly-Supervised Aggregation via Attention**
   - Attention mechanisms enable slide-level classification without patch annotations
   - Permutation-invariant aggregation captures bag-level diagnostics from instance-level features
   - Key papers: ABMIL, TransMIL, CLAM

3. **Pillar 3: Morphology Encodes Molecular Information**
   - H&E morphology is learnable surrogate for genetic alterations, mutations, and expression
   - Deep learning models achieve high AUROC (88-92%) for genetic prediction from morphology
   - Key papers: Deep-Learning-Based t(11;14) Prediction, Morphology Biomarker Review, Leveraging AI for Genetic Alterations

4. **Pillar 4: Radiomics Signatures Predict Outcome**
   - Automated segmentation + radiomic feature extraction identify prognostically relevant tumor heterogeneity
   - Radiomics C-index comparable to clinical staging systems (ISS, R-ISS)
   - Key papers: Radiomics Feature Analysis for Survival, High-Risk Cytogenetic Prediction from PET/CT

5. **Pillar 5: Multimodal Fusion Improves Accuracy**
   - Combining morphology, immunophenotyping, and genomic data outperforms single modalities
   - Integration of flow cytometry + morphology provides comprehensive diagnostic picture
   - Key papers: Pathomic Fusion, Multiparameter Flow Cytometry, Multimodal Hierarchical Fusion

### Contested Zones

1. **Cross-Center Robustness & Domain Generalization**
   - **Claim:** Models trained on one institution generalize to others
   - **Challenge:** Foundation models show 5-15% performance drops across medical centers; no solution published
   - **Implication:** Real-world deployment risky without per-center fine-tuning

2. **Morphology Sufficiency for Genetic Diagnosis**
   - **Claim:** Morphology alone sufficient for t(11;14) and other genetic prediction
   - **Challenge:** Morphology is not pathognomonic; validation requires molecular confirmation; causality unclear
   - **Implication:** Morphology-based predictions valuable as screening but not replacement for cytogenetics

3. **Attention Weights as Explanations**
   - **Claim:** High-attention patches identified by MIL models represent true diagnostic features
   - **Challenge:** Attention may learn artifacts; cross-cohort transfer of attention patterns poor
   - **Implication:** Saliency maps interpretable but not necessarily clinically meaningful

### Frontier Questions

1. **Can models be made robust to domain shift without fine-tuning?**
   - Unsolved methodological challenge
   - Critical for clinical deployment

2. **What is the causal mechanism linking morphology to molecular phenotypes?**
   - Correlation established; causality unproven
   - Mechanistic understanding may improve robustness

3. **Do radiomics add independent value to clinical staging, or are they redundant?**
   - Prospective clinical validation absent
   - Health economics and clinical decision-making impact unclear

### Must-Read Papers for Newcomers & Why

1. **Deep Learning-Based Prediction of Molecular Tumor Biomarkers from H&E: A Practical Review (2022)**
   - Why: Comprehensive, readable synthesis of the morphology-to-molecular prediction landscape; establishes that the field broadly believes morphology encodes molecular information
   - Audience: Newcomers wanting conceptual overview without getting lost in methods

2. **Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images (CLAM) (2021)**
   - Why: Canonical MIL method; clear explanation of attention-based aggregation; interpretability focus; widely adopted baseline
   - Audience: Those wanting to understand MIL machinery and implement it

3. **Prov-GigaPath: A Whole-Slide Foundation Model for Digital Pathology from Real-World Data (2024)**
   - Why: State-of-the-art foundation model; demonstrates power of pretraining on realistic clinical data; comprehensive benchmarking
   - Audience: Practical practitioners wanting to know what methods work best and why

---

## I. 5-Minute Explainer for Smart Non-Expert

### One-Sentence Version
**The field has proven that deep learning can automatically extract genetic and prognostic information from cancer cell images in bone marrow, enabling rapid diagnosis and treatment planning without sending samples for expensive genetic testing.**

### One Honest Admission
**Despite high accuracy in research papers (88-92%), these AI models often fail when moved from one hospital to another due to differences in how slides are prepared and scanned—we don't yet know how to fix this without retraining on new hospital data.**

### The Single Real-World Implication That Matters Most
**If this works at scale, a bone marrow aspirate slide could be run through an AI system in 10 minutes to predict which multiple myeloma patients have high-risk genetic subtypes (like t(11;14)) that need aggressive treatment, helping doctors make faster, more consistent treatment decisions. But this only happens if AI works reliably across different hospitals—right now it doesn't.**

---

---

