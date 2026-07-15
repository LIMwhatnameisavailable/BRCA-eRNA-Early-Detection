#!/usr/bin/env python3
"""
generate_clean_data.py
======================
Generate clean data CSVs from corrected data (correct CPM, hg38 realignment).

Output to outer_hg38/clean_data/:
  - dataset_{tcga,gse225846,gse229571}_final.csv (19 final features)
  - dataset_{tcga,gse225846,gse229571}_top300.csv (ANOVA Top300)
  - final_signature_list.csv (copied from results/)
  - Supplementary_Table_TOP300_Features.csv (copied from results/)

Does not retrain models, only loads data and extracts feature subsets.
Data loading logic is consistent with 1.1_nested_cv_pipeline.py.
"""

import pandas as pd
import numpy as np
import os
import sys
import random
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings('ignore')

# ============================================================
# Path configuration
# ============================================================
BASE_DIR = "../Data_Source"
RESULTS_DIR = os.path.join(BASE_DIR, "../results/clean_data_early_stage")
OUTPUT_DIR  = os.path.join(BASE_DIR, "../results/clean_data_early_stage")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "TCGA_Expr":        os.path.join(BASE_DIR, "TCGA/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt"),
    "TCGA_Normal_List": os.path.join(BASE_DIR, "TCGA/Normal_like_samples.csv"),
    "CLINICAL":         os.path.join(BASE_DIR, "TCGA/TCGA-BRCA.clinical.xlsx"),
    "GSE225846": {
        "name":      "GSE225846",
        "expr":      os.path.join(BASE_DIR, "GSE225846/counts_matrix_500bp_clean.txt"),
        "meta":      os.path.join(BASE_DIR, "GSE225846/SraRunTable_GSE225846.csv"),
        "stage_col": "stage",
        "type_col":  "type",
    },
    "GSE229571": {
        "name":      "GSE229571",
        "expr":      os.path.join(BASE_DIR, "GSE229571/counts_matrix_500bp_clean.txt"),
        "meta":      os.path.join(BASE_DIR, "GSE229571/SraRunTable_GSE229571.csv"),
        "stage_col": "tumor_stage",
        "type_col":  "tissue",
    },
}

TARGET_STAGES = [
    'Stage I', 'Stage IA', 'Stage IB',
    'Stage II', 'Stage IIA', 'Stage IIB',
    '1', '2', 'I', 'II'
]

ANOVA_K = 300


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_sample_class(row, type_col, stage_col):
    tissue_val = str(row[type_col]).lower().strip()
    if 'normal' in tissue_val or 'healthy' in tissue_val:
        return "Normal"
    if stage_col in row and pd.notna(row[stage_col]):
        stage_val = str(row[stage_col]).upper().strip().replace("STAGE", "").strip()
        if any(x in stage_val for x in ['III', 'IV', '3', '4']):
            return "Late"
        if any(x in stage_val for x in ['I', '1', '2']):
            return "Tumor"
    return "Other"


def load_tcga_data():
    """Load full TCGA data (identical to pipeline)"""
    print("\n>>> Loading TCGA data...")
    df = pd.read_csv(FILES["TCGA_Expr"], sep='\t', index_col=0)

    normal_df  = pd.read_csv(FILES["TCGA_Normal_List"], header=None, names=['SampleID'])
    target_ids = set([f"{str(x).strip()[:12]}_normal" for x in normal_df['SampleID']])
    final_normals = [c for c in df.columns if c in target_ids]

    try:
        pheno = pd.read_csv(FILES["CLINICAL"], sep='\t', encoding='utf-16')
    except Exception:
        pheno = pd.read_csv(FILES["CLINICAL"], sep='\t')

    stage_col = 'ajcc_pathologic_stage.diagnoses'
    if stage_col not in pheno.columns:
        stage_col = [c for c in pheno.columns
                     if 'stage' in c.lower() and 'diagnoses' in c.lower()][0]

    early_patients = set(pheno.loc[pheno[stage_col].isin(TARGET_STAGES), 'submitter_id'].values)
    valid_tumor    = [c for c in df.columns
                      if '_tumor' in c.lower() and c[:12] in early_patients]

    print(f"   Normal: {len(final_normals)}, Tumor: {len(valid_tumor)}")
    df_final = df[final_normals + valid_tumor].T
    labels   = pd.Series([0]*len(final_normals) + [1]*len(valid_tumor),
                         index=final_normals + valid_tumor)
    return df_final, labels


def load_external_data(config):
    """Load external GEO data, counts to CPM (1kb window, /1.0)"""
    name = config['name']
    print(f"\n>>> Loading external data {name}...")

    meta    = pd.read_csv(config['meta'])
    id_col  = 'Run' if 'Run' in meta.columns else meta.columns[0]
    valid_ids, labels_ext = [], []

    for _, row in meta.iterrows():
        cls = get_sample_class(row, config['type_col'], config['stage_col'])
        if cls in ["Normal", "Tumor"]:
            valid_ids.append(row[id_col])

    df = pd.read_csv(config['expr'], sep='\t', comment='#', index_col=0)
    df.columns = [c.replace('.bam', '').strip() for c in df.columns]
    df.columns = [c.split('/')[-1] if '/' in c else c for c in df.columns]
    drop_cols  = ['Chr', 'Start', 'End', 'Strand', 'Length']
    df         = df.drop(columns=[c for c in drop_cols if c in df.columns])

    common_ids = [x for x in valid_ids if x in df.columns]
    if not common_ids:
        print(f"   [WARN] {name} has no matching samples, returning None")
        return None, None

    # CPM: eRNA window +/-500bp = 1.0 kb
    df_counts = df[common_ids]
    df_cpm    = df_counts.div(df_counts.sum(axis=0), axis=1) * 1e6 / 1.0

    meta_idx = meta.set_index(id_col)
    labels_list = []
    for sid in common_ids:
        row = meta_idx.loc[sid]
        cls = get_sample_class(row, config['type_col'], config['stage_col'])
        labels_list.append(1 if cls == "Tumor" else 0)

    labels_series = pd.Series(labels_list, index=common_ids)
    print(f"   [{name}] Normal={labels_list.count(0)}, Tumor={labels_list.count(1)}")
    return df_cpm.T, labels_series


def get_expressed_features(df, threshold):
    if df is None:
        return set()
    return set((df > 0).mean(axis=0)[lambda s: s > threshold].index.tolist())


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":
    seed_everything(42)
    print("="*60)
    print("Generate Clean Data (corrected CPM, hg38 re-aligned)")
    print("="*60)

    # -- 1. Load data ---------------------------------------------
    tcga_X, tcga_y = load_tcga_data()
    gse1_X, gse1_y = load_external_data(FILES["GSE225846"])
    gse2_X, gse2_y = load_external_data(FILES["GSE229571"])

    if gse1_X is None or gse2_X is None:
        print("ERROR: External data loading failed")
        sys.exit(1)

    # -- 2. Three-dataset intersection filter --------------------
    print("\n>>> Three-dataset intersection filtering (technical feasibility)...")
    candidate_features = []
    for thresh in [0.5, 0.3, 0.2, 0.1]:
        f_tcga = get_expressed_features(tcga_X, thresh)
        f_gse1 = get_expressed_features(gse1_X, thresh)
        f_gse2 = get_expressed_features(gse2_X, thresh)
        common = list(f_tcga & f_gse1 & f_gse2)
        print(f"   Threshold >{thresh*100}%: intersection {len(common)} features")
        if len(common) >= 500:
            candidate_features = common
            print(f"   -> Using threshold {thresh}, candidate features: {len(candidate_features)}")
            break

    if not candidate_features:
        print("ERROR: Insufficient candidate features")
        sys.exit(1)

    # ── 3. ANOVA Top300 ─────────────────────────────────────
    print("\n>>> ANOVA Top300 selection (based on TCGA)...")
    X_full = tcga_X[candidate_features]
    y_full = tcga_y
    X_log = np.log1p(X_full.values)
    k = min(ANOVA_K, len(candidate_features))
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_log, y_full.values)
    top300_features = np.array(candidate_features)[selector.get_support()].tolist()
    top300_scores = selector.scores_[selector.get_support()]

    top300_df = pd.DataFrame({
        'eRNA_ID': top300_features,
        'ANOVA_F_Score': top300_scores
    }).sort_values('ANOVA_F_Score', ascending=False)
    print(f"   ANOVA Top{len(top300_features)} done")

    # -- 4. Load final feature signature -------------------------
    sig_path = os.path.join(RESULTS_DIR, "final_signature_list.csv")
    if not os.path.exists(sig_path):
        print(f"ERROR: {sig_path} not found, run 1.1_nested_cv_pipeline.py first")
        sys.exit(1)

    sig_df = pd.read_csv(sig_path)
    final_features = sig_df['Feature'].tolist()
    print(f"\n>>> Reading final features from pipeline results: {len(final_features)} features")

    # -- 5. Save CSVs --------------------------------------------
    print(f"\n>>> Saving to {OUTPUT_DIR}/")

    def save_dataset(X, y, features, name):
        """Extract feature subset from expression matrix, append target column"""
        df_out = X[features].copy()
        df_out['target'] = y
        path = os.path.join(OUTPUT_DIR, f"dataset_{name}.csv")
        df_out.to_csv(path)
        print(f"   ✓ dataset_{name}.csv  ({df_out.shape[0]} samples × {df_out.shape[1]-1} features)")
        return path

    # 5a. Final feature dataset
    save_dataset(tcga_X, tcga_y, final_features, "tcga_final")
    save_dataset(gse1_X, gse1_y, final_features, "gse225846_final")
    save_dataset(gse2_X, gse2_y, final_features, "gse229571_final")

    # 5b. Top300 dataset
    save_dataset(tcga_X, tcga_y, top300_features, "tcga_top300")
    save_dataset(gse1_X, gse1_y, top300_features, "gse225846_top300")
    save_dataset(gse2_X, gse2_y, top300_features, "gse229571_top300")

    # 5c. Copy signature file from results/
    import shutil
    shutil.copy2(sig_path, os.path.join(OUTPUT_DIR, "final_signature_list.csv"))
    print(f"   OK final_signature_list.csv (copied from results/)")

    # 5d. Save TOP300 supplementary table
    top300_df.to_csv(os.path.join(OUTPUT_DIR, "Supplementary_Table_TOP300_Features.csv"), index=False)
    print(f"   ✓ Supplementary_Table_TOP300_Features.csv")

    print(f"\n{'='*60}")
    print(f"All done! {len(os.listdir(OUTPUT_DIR))} files saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")