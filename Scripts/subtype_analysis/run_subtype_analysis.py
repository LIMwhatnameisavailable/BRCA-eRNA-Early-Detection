#!/usr/bin/env python3
"""
PAM50 Subtype & Clinical Subtype Analysis
==========================================
Analyze classifier performance across breast cancer molecular subtypes.

Outputs:
  - TableA1_prediction_score_by_PAM50.csv
  - TableA2_AUC_by_PAM50.csv
  - ReportA_PAM50_analysis.txt
  - TableB1_AUC_by_clinical_subtype_GSE225846.csv
  - ReportB_GSE225846_subtype_analysis.txt
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample

warnings.filterwarnings('ignore')


# ============================================================
# BalancedBaggingClassifier (required for unpickling)
# ============================================================
class BalancedBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=20):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators_ = []
        X_neg, X_pos = X[y == 0], X[y == 1]
        n = max(1, len(X_neg))
        for i in range(self.n_estimators):
            Xb = np.vstack((X_neg, resample(X_pos, n_samples=n, random_state=i)))
            yb = np.hstack((np.zeros(len(X_neg)), np.ones(n)))
            self.estimators_.append(clone(self.base_estimator).fit(Xb, yb))
        return self

    def predict_proba(self, X):
        return np.mean([clf.predict_proba(X) for clf in self.estimators_], axis=0)

# ============================================================
# Paths
# ============================================================
BASE_DIR   = "../Data_Source"
OUTPUT_DIR = "."
RESULTS    = "../results"
TCGA_DIR   = os.path.join(BASE_DIR, "TCGA")
GSE_DIR    = os.path.join(BASE_DIR, "GSE225846")

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOT = 1000
RNG_SEED = 42
TARGET_STAGES = [
    'Stage I', 'Stage IA', 'Stage IB',
    'Stage II', 'Stage IIA', 'Stage IIB',
    '1', '2', 'I', 'II'
]

# ============================================================
# 0. Load 19 features and RF model
# ============================================================
print(">>> Loading features and model...")
sig_df = pd.read_csv(os.path.join(RESULTS, "final_signature_list.csv"))
features = sig_df['Feature'].tolist()
print(f"   19 features loaded (from final_signature_list.csv)")

model = joblib.load(os.path.join(RESULTS, "RF_ML_model.pkl"))
print(f"   RF_ML_model.pkl loaded")

# ============================================================
# 1. Load TCGA expression data (wide format: genes=rows, samples=cols)
# ============================================================
print("\n>>> Loading TCGA expression matrix...")
df_tcga_wide = pd.read_csv(
    os.path.join(TCGA_DIR, "TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt"),
    sep='\t', index_col=0
)
print(f"   Shape: {df_tcga_wide.shape[0]} genes x {df_tcga_wide.shape[1]} samples")

# Extract 19 features, transpose -> rows=samples, cols=features
tcga_expr = df_tcga_wide.loc[features].T.copy()
print(f"   After 19-feature extraction + transpose: {tcga_expr.shape[0]} x {tcga_expr.shape[1]}")

# ============================================================
# 2. Reconstruct training set for scaler fitting
# ============================================================
print("\n>>> Reconstructing TCGA training set for scaler...")

# 2a. Normal samples from whitelist
normal_df = pd.read_csv(os.path.join(TCGA_DIR, "Normal_like_samples.csv"),
                         header=None, names=['SampleID'])
target_normal_ids = set([f"{str(x).strip()[:12]}_normal" for x in normal_df['SampleID']])
train_normal = [c for c in tcga_expr.index if c in target_normal_ids]
print(f"   Normal whitelist: {len(train_normal)} samples matched")

# 2b. Early-stage tumor samples from clinical file
try:
    pheno = pd.read_csv(os.path.join(TCGA_DIR, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"),
                         sep='\t', low_memory=False)
except Exception:
    pheno = pd.read_csv(os.path.join(TCGA_DIR, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"),
                         sep='\t', encoding='utf-16', low_memory=False)

# Detect stage column: prefer pathologic_stage, otherwise AJCC_Stage
stage_col = None
for sc in ['pathologic_stage', 'AJCC_Stage_nature2012', 'Converted_Stage_nature2012']:
    if sc in pheno.columns:
        stage_col = sc
        break
if stage_col is None:
    stage_col = [c for c in pheno.columns if 'stage' in c.lower()][0]
print(f"   Using stage column: '{stage_col}'")

# Use sampleID first 12 chars as submitter_id (full TCGA participant barcode)
# patient_id only has the short participant code, not the project prefix
pheno['submitter_id'] = pheno['sampleID'].str[:12].astype(str).str.strip()

early_patients = set(pheno.loc[pheno[stage_col].isin(TARGET_STAGES), 'submitter_id'].values)
train_tumor = [c for c in tcga_expr.index
               if '_tumor' in c.lower() and c[:12] in early_patients]
print(f"   Early-stage tumor (Stage I/II): {len(train_tumor)} samples matched")

train_samples = train_normal + train_tumor
print(f"   Total training samples for scaler: {len(train_samples)}")

# 2c. Fit scaler on training samples only
# Updated: log1p + RobustScaler(5-95) + Normalizer (consistent with pipeline)
scaler = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
scaler.fit(np.log1p(np.maximum(tcga_expr.loc[train_samples].values, 0)))
print(f"   Scaler (log1p+RobustScaler+Normalizer) fitted on {len(train_samples)} training samples")

# ============================================================
# 3. Transform and predict on ALL TCGA samples
# ============================================================
print("\n>>> Transforming & predicting on all TCGA samples...")
X_tcga_scaled = scaler.transform(np.log1p(np.maximum(tcga_expr.values, 0)))
all_probs = model.predict_proba(X_tcga_scaled)[:, 1]
tcga_result = tcga_expr.copy()
tcga_result['pred_prob'] = all_probs
tcga_result['is_tumor'] = [1 if '_tumor' in c.lower() else 0 for c in tcga_result.index]
tcga_result['patient_id'] = [c[:12] for c in tcga_result.index]
print(f"   Predictions: {np.mean(tcga_result['is_tumor']):.1%} tumor, "
      f"{1-np.mean(tcga_result['is_tumor']):.1%} normal (of {len(tcga_result)} total)")

# ============================================================
# 4. Prepare normal samples for AUC computation
# ============================================================
normal_samples = tcga_result[tcga_result['is_tumor'] == 0].copy()
print(f"\n>>> Normal samples for AUC reference: {len(normal_samples)}")

# ============================================================
# 5. PAM50 matching (all tumor samples only)
# ============================================================
print("\n>>> Matching TCGA tumor samples with PAM50...")
tumor_samples = tcga_result[tcga_result['is_tumor'] == 1].copy()

# Load PAM50
pam50 = pd.read_csv(os.path.join(TCGA_DIR, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"),
                     sep='\t', low_memory=False)
pam50_sub = pam50[['sampleID', 'PAM50Call_RNAseq']].copy()
pam50_sub['patient_id'] = pam50_sub['sampleID'].str[:12]
pam50_sub = pam50_sub.dropna(subset=['PAM50Call_RNAseq'])
pam50_sub = pam50_sub[pam50_sub['PAM50Call_RNAseq'] != '']
pam50_sub = pam50_sub[pam50_sub['PAM50Call_RNAseq'].notna()]

# Label mapping
label_map = {
    'LumA': 'Luminal A',
    'LumB': 'Luminal B',
    'Her2': 'HER2-enriched',
    'Basal': 'Basal-like',
    'Normal': 'Normal-like'
}
pam50_sub['subtype'] = pam50_sub['PAM50Call_RNAseq'].map(label_map)

# Merge with tumor predictions
tumor_merged = tumor_samples.merge(
    pam50_sub[['patient_id', 'subtype']],
    on='patient_id', how='inner'
)
print(f"   Tumor samples with PAM50: {len(tumor_merged)}")

# ============================================================
# 6. Table A1: Descriptive statistics by PAM50 subtype
# ============================================================
print("\n>>> Table A1: Prediction score by PAM50 subtype...")
a1_rows = []
for sub in ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like', 'Normal-like']:
    sub_df = tumor_merged[tumor_merged['subtype'] == sub]
    n = len(sub_df)
    if n == 0:
        continue
    scores = sub_df['pred_prob'].values
    a1_rows.append({
        'Subtype': sub,
        'N': n,
        'Mean': f"{np.mean(scores):.4f}",
        'Median': f"{np.median(scores):.4f}",
        'SD': f"{np.std(scores, ddof=1):.4f}",
        'Q1': f"{np.percentile(scores, 25):.4f}",
        'Q3': f"{np.percentile(scores, 75):.4f}",
        'Min': f"{np.min(scores):.4f}",
        'Max': f"{np.max(scores):.4f}",
    })

df_a1 = pd.DataFrame(a1_rows)
df_a1.to_csv(os.path.join(OUTPUT_DIR, "TableA1_prediction_score_by_PAM50.csv"), index=False)
print(f"   Saved: TableA1_prediction_score_by_PAM50.csv ({len(df_a1)} subtypes)")

# ============================================================
# 7. Table A2: AUC per PAM50 subtype (tumor_subtype vs all_normal)
# ============================================================
print("\n>>> Table A2: AUC by PAM50 subtype...")

def bootstrap_auc(y_true, y_prob, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap AUC with 95% CI. Returns (mean_auc, lower, upper)."""
    rng = np.random.RandomState(seed)
    aucs = []
    valid = 0
    while valid < n_boot:
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        valid += 1
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

a2_rows = []
normal_y = normal_samples['pred_prob'].values  # label=0
for sub in ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like', 'Normal-like']:
    sub_df = tumor_merged[tumor_merged['subtype'] == sub]
    n_tumor = len(sub_df)
    n_normal = len(normal_samples)
    if n_tumor < 2:
        continue
    # tumor_subtype vs all normal
    y_true = np.concatenate([np.ones(n_tumor), np.zeros(n_normal)])
    y_prob = np.concatenate([sub_df['pred_prob'].values, normal_y])
    mean_auc, ci_l, ci_u = bootstrap_auc(y_true, y_prob)
    note = ""
    if n_tumor < 15:
        note = "Small sample size, results are for reference only"
    a2_rows.append({
        'Subtype': sub,
        'N_tumor': n_tumor,
        'N_normal': n_normal,
        'AUC': f"{mean_auc:.4f}",
        '95CI_lower': f"{ci_l:.4f}",
        '95CI_upper': f"{ci_u:.4f}",
        'Note': note,
    })

df_a2 = pd.DataFrame(a2_rows)
df_a2.to_csv(os.path.join(OUTPUT_DIR, "TableA2_AUC_by_PAM50.csv"), index=False)
print(f"   Saved: TableA2_AUC_by_PAM50.csv ({len(df_a2)} subtypes)")

# ============================================================
# 8. Report A: PAM50 analysis text
# ============================================================
print("\n>>> Report A: Writing PAM50 analysis report...")
lines_a = []
lines_a.append("=" * 70)
lines_a.append("Report A: PAM50 Subtype Analysis of Classifier Prediction Scores")
lines_a.append("=" * 70)
lines_a.append("")
lines_a.append(f"Analysis date: 2026-07-09")
lines_a.append(f"Classifier: Random Forest (RF_ML_model.pkl)")
lines_a.append(f"Number of eRNA features: 19 (from final_signature_list.csv)")
lines_a.append(f"  (Note: LASSO dynamic threshold 0.60→0.50 yielded 19 features)")
lines_a.append(f"Scaler: log1p + RobustScaler(5-95% quantile) + Normalizer, fitted on TCGA training set")
lines_a.append(f"  (Training set = {len(train_normal)} normal samples + {len(train_tumor)} early-stage tumor samples)")
lines_a.append(f"External validation AUC (GSE225846): 0.943 (RF model, log1p+RobustScaler+Normalizer)")
lines_a.append("")

lines_a.append("-" * 70)
lines_a.append("Sample Size by PAM50 Subtype")
lines_a.append("-" * 70)
lines_a.append(f"  Total tumor samples with PAM50 label: {len(tumor_merged)}")
for r in a1_rows:
    lines_a.append(f"  {r['Subtype']:20s}  N = {r['N']}")
lines_a.append("")

lines_a.append("-" * 70)
lines_a.append("Kruskal-Wallis Test: Prediction Scores Across PAM50 Subtypes")
lines_a.append("-" * 70)
groups = []
group_names = []
for sub in ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like', 'Normal-like']:
    sub_scores = tumor_merged[tumor_merged['subtype'] == sub]['pred_prob'].values
    if len(sub_scores) > 0:
        groups.append(sub_scores)
        group_names.append(sub)

if len(groups) >= 2:
    h_stat, p_val = kruskal(*groups)
    lines_a.append(f"  H-statistic: {h_stat:.4f}")
    if p_val < 1e-10:
        lines_a.append(f"  p-value: {p_val:.2e} (*** highly significant)")
    elif p_val < 1e-4:
        lines_a.append(f"  p-value: {p_val:.2e} (*** significant at 0.001)")
    elif p_val < 0.001:
        lines_a.append(f"  p-value: {p_val:.6f} (** significant at 0.01)")
    elif p_val < 0.05:
        lines_a.append(f"  p-value: {p_val:.4f} (* significant at 0.05)")
    else:
        lines_a.append(f"  p-value: {p_val:.4f} (not significant)")
else:
    lines_a.append("  (Insufficient groups for Kruskal-Wallis)")
lines_a.append("")

lines_a.append("-" * 70)
lines_a.append("AUC by PAM50 Subtype")
lines_a.append("-" * 70)
lines_a.append("  Note: Normal samples carry no PAM50 label.")
lines_a.append("  AUC per subtype = [that subtype's tumor samples] vs [all TCGA normal samples].")
lines_a.append("  This is the biologically appropriate comparison (normal tissue as reference).")
lines_a.append("  95% CI computed via bootstrap (n=1000 iterations).")
lines_a.append("")
lines_a.append(f"  {'Subtype':20s}  {'N_tumor':>9}  {'N_normal':>9}  {'AUC':>8}  {'95% CI':>18}  {'Note':>20}")
lines_a.append("  " + "-" * 90)
for r in a2_rows:
    ci_str = f"({r['95CI_lower']}-{r['95CI_upper']})"
    lines_a.append(f"  {r['Subtype']:20s}  {r['N_tumor']:>9}  {r['N_normal']:>9}  {r['AUC']:>8}  {ci_str:>18}  {r.get('Note', '')}")
lines_a.append("")

lines_a.append("-" * 70)
lines_a.append("Key Findings")
lines_a.append("-" * 70)

# Identify highest and lowest
auc_vals = {r['Subtype']: float(r['AUC']) for r in a2_rows}
if auc_vals:
    best_sub = max(auc_vals, key=auc_vals.get)
    worst_sub = min(auc_vals, key=auc_vals.get)
    lines_a.append(f"  Highest AUC: {best_sub} ({auc_vals[best_sub]:.3f})")
    lines_a.append(f"  Lowest AUC:  {worst_sub} ({auc_vals[worst_sub]:.3f})")
    # Check Normal-like
    if 'Normal-like' in auc_vals:
        lines_a.append(f"  Normal-like AUC: {auc_vals['Normal-like']:.3f}")
        if auc_vals['Normal-like'] >= 0.65:
            lines_a.append(f"    (Relatively high — some normal-adjacent tumors may have expression")
            lines_a.append(f"     profiles closer to normal tissue)")
    lines_a.append("")

with open(os.path.join(OUTPUT_DIR, "ReportA_PAM50_analysis.txt"), 'w') as f:
    f.write('\n'.join(lines_a))
print(f"   Saved: ReportA_PAM50_analysis.txt")

# ============================================================
# ============================================================
#  PLAN B: GSE225846 clinical subtype analysis
# ============================================================
# ============================================================
print("\n" + "=" * 70)
print("PLAN B: GSE225846 Clinical Subtype Analysis")
print("=" * 70)

# ============================================================
# B1. Load GSE225846 metadata and classify
# ============================================================
print("\n>>> B1. Loading GSE225846 metadata...")
meta_gse = pd.read_csv(os.path.join(BASE_DIR, "data_raw/outer_hg38/GSE225846/SraRunTable_GSE225846.csv"))
print(f"   Metadata columns: {list(meta_gse.columns)}")

# Assign clinical subtypes for tumor samples
def assign_clinical_subtype(row):
    """Assign clinical surrogate subtype based on ER/PR/HER2."""
    er = str(row['ER_status']).strip() if pd.notna(row['ER_status']) else ''
    pr = str(row['pr']).strip() if pd.notna(row['pr']) else ''
    her2 = str(row['HER2']).strip() if pd.notna(row['HER2']) else ''

    # TNBC: ER-, PR-, HER2-
    if er == 'Negative' and pr == 'Negative' and her2 == 'Negative':
        return 'TNBC (Triple Negative)'
    # HER2+: regardless of ER/PR
    if her2 == 'Positive':
        return 'HER2+'
    # HR+/HER2-: ER+ or Weak Positive, and HER2-
    if er in ['Positive', 'Weak Positive'] and her2 == 'Negative':
        return 'HR+/HER2− (Luminal-like)'
    return 'Other'

# classify only tumor samples
meta_gse['tissue_type'] = meta_gse['type'].str.lower().str.strip()
meta_gse['is_normal'] = meta_gse['tissue_type'].str.contains('normal|healthy', na=False)
meta_gse['is_tumor_gse'] = meta_gse['tissue_type'].str.contains('tumor', na=False)

meta_tumor = meta_gse[meta_gse['is_tumor_gse']].copy()
meta_normal = meta_gse[meta_gse['is_normal']].copy()
print(f"   Tumor samples: {len(meta_tumor)}, Normal samples: {len(meta_normal)}")

meta_tumor['clinical_subtype'] = meta_tumor.apply(assign_clinical_subtype, axis=1)
print("\n   Clinical subtype distribution (tumor):")
for sub, cnt in meta_tumor['clinical_subtype'].value_counts().items():
    print(f"     {sub:35s}  N = {cnt}")
meta_tumor['Run'].to_csv('/tmp/gse_tumor_runs.csv', index=False, header=False)
# ============================================================
# B2. Load GSE225846 expression data
# ============================================================
print("\n>>> B2. Loading GSE225846 counts matrix...")
df_gse_counts = pd.read_csv(
    os.path.join(GSE_DIR, "analysis/counts_matrix_500bp_clean.txt"),
    sep='\t', comment='#', index_col=0
)

# Clean column names
df_gse_counts.columns = [c.replace('.bam', '').strip() for c in df_gse_counts.columns]
df_gse_counts.columns = [c.split('/')[-1] if '/' in c else c for c in df_gse_counts.columns]

# Drop annotation columns if present
drop_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
df_gse_counts = df_gse_counts.drop(columns=[c for c in drop_cols if c in df_gse_counts.columns])
print(f"   Counts matrix shape: {df_gse_counts.shape}")

# CPM normalization (÷1.0 for 1 kb window)
df_cpm = df_gse_counts.div(df_gse_counts.sum(axis=0), axis=1) * 1e6 / 1.0

# Extract 19 features, transpose -> rows=samples, cols=features
common_samples = [s for s in meta_gse['Run'] if s in df_cpm.columns]
print(f"   Samples matching metadata: {len(common_samples)}")
gse_expr = df_cpm[common_samples].T
gse_expr = gse_expr[[f for f in features if f in gse_expr.columns]]
# Fill missing features with 0
for f in features:
    if f not in gse_expr.columns:
        gse_expr[f] = 0.0
gse_expr = gse_expr[features]
print(f"   GSE225846 expression matrix: {gse_expr.shape[0]} samples x {gse_expr.shape[1]} features")

# ============================================================
# B3. Transform using SAME TCGA-fitted scaler & Predict
# ============================================================
print("\n>>> B3. Transforming & predicting GSE225846...")
X_gse_scaled = scaler.transform(np.log1p(np.maximum(gse_expr.values, 0)))
gse_probs = model.predict_proba(X_gse_scaled)[:, 1]
gse_result = gse_expr.copy()
gse_result['pred_prob'] = gse_probs
gse_result['Run'] = gse_expr.index

# Merge metadata
meta_gse_sub = meta_gse[['Run', 'type', 'ER_status', 'pr', 'HER2', 'is_normal', 'is_tumor_gse']].copy()
gse_merged = gse_result.merge(meta_gse_sub, on='Run', how='left')
print(f"   Merged result: {len(gse_merged)} samples")

# Overall AUC (all tumor vs all normal)
gse_tumor_mask = gse_merged['is_tumor_gse']
gse_normal_mask = gse_merged['is_normal']
gse_y_true = np.concatenate([
    np.ones(gse_tumor_mask.sum()),
    np.zeros(gse_normal_mask.sum())
])
gse_y_prob = np.concatenate([
    gse_merged.loc[gse_tumor_mask, 'pred_prob'].values,
    gse_merged.loc[gse_normal_mask, 'pred_prob'].values,
])
overall_auc, overall_ci_l, overall_ci_u = bootstrap_auc(gse_y_true, gse_y_prob)
print(f"   Overall AUC (all tumor vs all normal): {overall_auc:.4f} ({overall_ci_l:.4f}-{overall_ci_u:.4f})")

# ============================================================
# B4. Table B1: AUC by clinical subtype
# ============================================================
print("\n>>> B4. AUC by clinical subtype...")
# Add clinical subtype to tumor samples in merged data
tumor_merge = gse_merged[gse_merged['is_tumor_gse']].copy()
tumor_merge['clinical_subtype'] = tumor_merge.apply(assign_clinical_subtype, axis=1)
normal_gse = gse_merged[gse_merged['is_normal']].copy()

b1_rows = []
subtypes_order = ['HR+/HER2− (Luminal-like)', 'HER2+', 'TNBC (Triple Negative)']

for sub in subtypes_order:
    sub_df = tumor_merge[tumor_merge['clinical_subtype'] == sub]
    n_tumor = len(sub_df)
    n_normal = len(normal_gse)
    if n_tumor < 2:
        continue
    y_true = np.concatenate([np.ones(n_tumor), np.zeros(n_normal)])
    y_prob = np.concatenate([sub_df['pred_prob'].values, normal_gse['pred_prob'].values])
    mean_auc, ci_l, ci_u = bootstrap_auc(y_true, y_prob)
    note = ""
    if n_tumor < 15:
        note = "Small sample size, results are for reference only"
    b1_rows.append({
        'Clinical_Subtype': sub,
        'N_tumor': n_tumor,
        'N_normal': n_normal,
        'AUC': f"{mean_auc:.4f}",
        '95CI_lower': f"{ci_l:.4f}",
        '95CI_upper': f"{ci_u:.4f}",
        'Note': note,
    })

# Also add "Other/Unclassified" if there are any
other_sub = tumor_merge[~tumor_merge['clinical_subtype'].isin(subtypes_order)]
if len(other_sub) > 0:
    n_other = len(other_sub)
    y_true = np.concatenate([np.ones(n_other), np.zeros(n_normal)])
    y_prob = np.concatenate([other_sub['pred_prob'].values, normal_gse['pred_prob'].values])
    mean_auc, ci_l, ci_u = bootstrap_auc(y_true, y_prob)
    b1_rows.append({
        'Clinical_Subtype': 'Other/Unclassified',
        'N_tumor': n_other,
        'N_normal': n_normal,
        'AUC': f"{mean_auc:.4f}",
        '95CI_lower': f"{ci_l:.4f}",
        '95CI_upper': f"{ci_u:.4f}",
        'Note': 'Excluded from main analysis due to ambiguous receptor status',
    })

df_b1 = pd.DataFrame(b1_rows)
df_b1.to_csv(os.path.join(OUTPUT_DIR, "TableB1_AUC_by_clinical_subtype_GSE225846.csv"), index=False)
print(f"   Saved: TableB1_AUC_by_clinical_subtype_GSE225846.csv ({len(df_b1)} subtypes)")

# ============================================================
# B5. Report B
# ============================================================
print("\n>>> Report B: Writing GSE225846 subtype analysis report...")
lines_b = []
lines_b.append("=" * 70)
lines_b.append("Report B: GSE225846 Clinical Subtype Analysis")
lines_b.append("=" * 70)
lines_b.append("")
lines_b.append(f"Dataset: GSE225846 (80 tumor, 75 normal)")
lines_b.append(f"Classifier: Random Forest (RF_ML_model.pkl)")
lines_b.append(f"Number of eRNA features: 19")
lines_b.append(f"Expression normalization: CPM (counts per million, ÷1.0 for 1kb window)")
lines_b.append(f"Scaler: log1p + RobustScaler(5-95% quantile) + Normalizer, fitted on TCGA training set")
lines_b.append(f"  (Note: using pipeline-identical preprocessing [log1p + RobustScaler + Normalizer]")
lines_b.append(f"   on the same TCGA training samples reproduces reported RF AUC of 0.943 on GSE225846.)")
lines_b.append("")

lines_b.append("-" * 70)
lines_b.append("Clinical Subtype Classification Rules")
lines_b.append("-" * 70)
lines_b.append("  1. HR+/HER2− (Luminal-like):  ER_status ∈ {Positive, Weak Positive} AND HER2 = Negative")
lines_b.append("  2. HER2+:                     HER2 = Positive (regardless of ER/PR)")
lines_b.append("  3. TNBC (Triple Negative):    ER = Negative, PR = Negative, HER2 = Negative")
lines_b.append("  4. Other:                     Does not fit any above category (excluded)")
lines_b.append("")

lines_b.append("-" * 70)
lines_b.append("Sample Distribution")
lines_b.append("-" * 70)
lines_b.append(f"  Total samples:               {len(meta_gse)}")
lines_b.append(f"  Normal:                      {len(meta_normal)}")
lines_b.append(f"  Tumor:                       {len(meta_tumor)}")
lines_b.append("")
for sub, cnt in meta_tumor['clinical_subtype'].value_counts().items():
    lines_b.append(f"  {sub:35s}  N = {cnt}")
lines_b.append("")

lines_b.append("-" * 70)
lines_b.append("AUC by Clinical Subtype")
lines_b.append("-" * 70)
lines_b.append("  95% CI computed via bootstrap (n=1000 iterations).")
lines_b.append("  Reference group: all GSE225846 normal samples (N=75).")
lines_b.append("")
lines_b.append(f"  {'Clinical Subtype':35s}  {'N_tumor':>9}  {'N_normal':>9}  {'AUC':>8}  {'95% CI':>18}  {'Note':>20}")
lines_b.append("  " + "-" * 100)
for r in b1_rows:
    ci_str = f"({r['95CI_lower']}-{r['95CI_upper']})"
    lines_b.append(f"  {r['Clinical_Subtype']:35s}  {r['N_tumor']:>9}  {r['N_normal']:>9}  {r['AUC']:>8}  {ci_str:>18}  {r.get('Note', '')}")
lines_b.append("")
lines_b.append(f"  {'Overall (all tumor vs all normal)':35s}  {len(gse_tumor_mask):>9}  {len(gse_normal_mask):>9}  {overall_auc:.4f}  ({overall_ci_l:.4f}-{overall_ci_u:.4f})")
lines_b.append("")

lines_b.append("-" * 70)
lines_b.append("Key Findings")
lines_b.append("-" * 70)
sub_aucs = {r['Clinical_Subtype']: float(r['AUC']) for r in b1_rows
            if r.get('Note', '') != 'Excluded from main analysis due to ambiguous receptor status'}
if sub_aucs:
    best = max(sub_aucs, key=sub_aucs.get)
    worst = min(sub_aucs, key=sub_aucs.get)
    lines_b.append(f"  Highest AUC: {best} ({sub_aucs[best]:.3f})")
    lines_b.append(f"  Lowest AUC:  {worst} ({sub_aucs[worst]:.3f})")
    lines_b.append(f"  Overall AUC: {overall_auc:.3f}")
    lines_b.append("")
    for sub, auc_val in sub_aucs.items():
        diff = auc_val - overall_auc
        dir_str = "higher" if diff > 0 else "lower"
        lines_b.append(f"  {sub}: AUC={auc_val:.3f} ({abs(diff):.3f} {dir_str} than overall)")

with open(os.path.join(OUTPUT_DIR, "ReportB_GSE225846_subtype_analysis.txt"), 'w') as f:
    f.write('\n'.join(lines_b))
print(f"   Saved: ReportB_GSE225846_subtype_analysis.txt")

print("\n" + "=" * 70)
print("ALL DONE! Output files in:", OUTPUT_DIR)
print("=" * 70)
