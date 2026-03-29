# TCGA-BRCA eRNA Early-Stage Discrimination Project

## 📌 Project Overview
This repository contains the complete Python-based machine learning and deep learning pipeline for the early-stage discrimination of Breast Cancer (BRCA) using enhancer RNA (eRNA) profiles. 

As a parallel extension to our Prognostic Signature Project, this module rigorously defines a "normal-like" reference to mitigate field cancerization, quantifies eRNA expression from raw RNA-seq data of independent cohorts (GSE225846, GSE229571), and evaluates 10 distinct classification algorithms. The pipeline features a high-performance 96-core parallel quantification script, strict data leakage prevention during preprocessing, and an advanced Attention-1D-CNN model.

## 📂 Repository Structure
The repository is designed as a standalone project, organized chronologically from raw data quantification to model bootstrap evaluation.
*Note: The `Data_Source` folder is excluded from the repository due to size limitations.*

```text
Early_Discrimination_Python/
├── README.md                           <-- Project documentation
├── Data_Source/                        <-- Required input files (See Data Preparation)
├── Scripts/                            
│   ├── 0_High_Performance_Quant_Pipeline.sh  <-- HISAT2 & featureCounts pipeline
│   ├── 1_Feature_Selection.py                <-- ANOVA & Bootstrap LASSO for 18-eRNA signature
│   ├── 2_Data_Preparation.py                 <-- Strict target matching & dataset generation
│   └── 3_Model_Comparison_Bootstrap.py       <-- Log1p-Robust scaling, DL/ML training, and 5000x Bootstrap
└── results                                   <-- Automatically generated ROC curves, models (.pkl), and metrics
```

## 💾 Data Preparation (Crucial)
⚠️ **Action Required:** Raw data is NOT included in this repository. Please create a folder named `Data_Source` and organize the files exactly as required by the Python scripts:

| File Category | Required Filename / Path | Description |
| :--- | :--- | :--- |
| **TCGA eRNA Expr** | `TCGA/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt` | Processed from TCeA |
| **TCGA Normal Ref** | `TCGA/Normal_like_samples.csv` | *Generated via Prognosis R Script 8* |
| **TCGA Clinical** | `TCGA-BRCA.clinical.xlsx` | Phenotype data for stage filtering |
| **GEO Metadata** | `GSE225846/SraRunTable_GSE225846.csv` | Clinical metadata for external cohort 1 |
| **GEO Metadata** | `GSE229571/SraRunTable_GSE229571.csv` | Clinical metadata for external cohort 2 |
| **SAF Annotation** | `TCGA/eRNA_standard_500bp.saf` | Custom SAF file for eRNA quantification |

> 🔗 **Important Link to Prognosis Module:** 
> The `Normal_like_samples.csv` used here excludes field cancerization artifacts. The R code to generate this reference is located in the companion Prognosis repository: 
> 👉 **`Prognosis_R/R_Scripts/8_Quality_Control_for_Normal_Like_References.R`**

## 🚀 How to Run the Pipeline

### Part I: High-Performance Data Quantification
**Script 0: `0_High_Performance_Quant_Pipeline.sh`**
*   **Function:** A Bash script utilizing FIFO named pipes for concurrency control. It aligns raw FASTQ files using `HISAT2`, pipes directly to `samtools sort` to save I/O, and performs multi-threaded quantification using `featureCounts` against the custom 500bp SAF file.
*   **Output:** Generates `counts_matrix_500bp_clean.txt` for each GSE dataset.

### Part II: Feature Engineering & Dataset Construction
**Script 1: `1_Feature_Selection.py`**
*   **Function:** Intersects highly expressed eRNAs across TCGA and GEO cohorts. Applies univariate ANOVA F-statistics to select the top 300 eRNAs, followed by a 100-iteration Bootstrap LASSO regression to define a robust 18-eRNA diagnostic signature.

**Script 2: `2_Data_Preparation.py`**
*   **Function:** Strictly filters patients by early-stage criteria (Stage I/II). Matches the selected 18-eRNA and top-300 eRNA features across all cohorts, converting raw counts to RPKM for the GEO datasets, and outputs finalized CSV datasets for model training.

### Part III: Model Training & Rigorous Evaluation
**Script 3: `3_Model_Comparison_Bootstrap.py`**
*   **Function:** The core evaluation engine. 
    *   **Preprocessing:** Implements strict anti-leakage `Log1p + Global Robust Scaling` (fit on train, transform on test).
    *   **Modeling:** Trains 8 traditional ML models and 2 DL models (including a custom `Attention-1D-CNN` with GroupNorm and SE-Blocks). Handles class imbalance via `BalancedBaggingClassifier`.
    *   **Validation:** Evaluates generalization on GSE225846 and GSE229571 using 5,000 bootstrap iterations. 
    *   **Output:** Generates smoothed ROC curves, exports trained models as `.pkl` files (ready for the Shiny Web App), and compiles a publication-ready metrics table (AUC, Sensitivity, Specificity, F1) with 95% CIs.

## 🛠 Dependencies
*   **Python Environment:** Python 3.9+, `scikit-learn` (1.6.1), `PyTorch` (2.3.1), `joblib`, `numpy`, `pandas`, `matplotlib`.
*   **Bioinformatics Tools:** `HISAT2`, `samtools`, `featureCounts` (Subread package).
