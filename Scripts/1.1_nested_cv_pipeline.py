"""
Suggestion 1.1 - Nested Cross-Validation Pipeline
====================================================
Revision notes:
1. CPM fix: external data / 1.0 (corrected from original / 0.5 error)
2. Three-dataset intersection as global technical feasibility filter
3. Added 10-fold Nested CV: ANOVA + Scaler + Bootstrap LASSO
   Strictly confined within each outer training fold
4. Final model training logic unchanged (full TCGA -> external validation)
5. New figure: Nested CV model AUC distribution

Output: ./results/
Does not overwrite original results (in ../results/step3_optimized/)

Author: Major Revision Suggestion 1.1
"""

import os
import pandas as pd
import numpy as np
import sys
import random
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from joblib import Parallel, delayed
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed, skipping XGB model")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed, skipping CNN model")

warnings.filterwarnings('ignore')

# ============================================================
# Path configuration
# ============================================================
BASE_DIR = "../Data_Source"
OUTPUT_DIR = "../results/clean_data_early_stage"
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

N_OUTER_FOLDS   = 10
N_LASSO_BOOT    = 100
LASSO_ALPHA     = 0.015
LASSO_FREQ_THR  = 0.60
ANOVA_K         = 300
N_BOOTSTRAPS    = 5000
if HAS_TORCH:
    torch.set_num_threads(1)

N_JOBS          = 36   # Max 36 cores (server has 96, reserving for other users)

# ============================================================
# Utility functions
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def compute_single_metrics(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_thr = thresholds[np.argmax(tpr - fpr)]
    y_pred   = (y_prob >= best_thr).astype(int)
    auc_val  = auc(fpr, tpr)
    acc      = accuracy_score(y_true, y_pred)
    sens     = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec     = precision_score(y_true, y_pred, zero_division=0)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    return [auc_val, acc, sens, spec, prec, f1]


def evaluate_with_bootstrap(y_true, y_prob, n_bootstraps=N_BOOTSTRAPS, seed=42):
    rng = np.random.RandomState(seed)
    metrics_list = []
    valid_iters  = 0
    while valid_iters < n_bootstraps:
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics_list.append(compute_single_metrics(y_true[idx], y_prob[idx]))
        valid_iters += 1
    arr   = np.array(metrics_list)
    means = np.mean(arr, axis=0)
    lower = np.percentile(arr, 2.5,  axis=0)
    upper = np.percentile(arr, 97.5, axis=0)
    fmt   = [f"{m:.3f} ({l:.3f}-{u:.3f})" for m, l, u in zip(means, lower, upper)]
    return fmt, means[0]

# ============================================================
# Data loading
# ============================================================

def load_tcga_data():
    """Load full TCGA data (all stage tumors + whitelist normal samples)"""
    print("\n>>> Loading TCGA data...")
    df = pd.read_csv(FILES["TCGA_Expr"], sep='\t', index_col=0)

    # Whitelist normal samples
    normal_df  = pd.read_csv(FILES["TCGA_Normal_List"], header=None, names=['SampleID'])
    target_ids = set([f"{str(x).strip()[:12]}_normal" for x in normal_df['SampleID']])
    final_normals = [c for c in df.columns if c in target_ids]

    # Early-stage tumor samples
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
    """
    Load external GEO dataset, counts to CPM (1kb window, corrected: /1.0)
    Note: original code incorrectly divided by 0.5, corrected to /1.0
    """
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
    # Strip full paths from column names (e.g., /mnt/.../SRR23588016.bam)
    df.columns = [c.split('/')[-1] if '/' in c else c for c in df.columns]
    drop_cols  = ['Chr', 'Start', 'End', 'Strand', 'Length']
    df         = df.drop(columns=[c for c in drop_cols if c in df.columns])

    common_ids = [x for x in valid_ids if x in df.columns]
    if not common_ids:
        print(f"   [WARN] {name} has no matching samples, returning None")
        return None, None

    # CPM correction: eRNA window +/-500bp = 1.0kb, divide by 1.0 (original incorrectly used 0.5)
    df_counts = df[common_ids]
    df_cpm    = df_counts.div(df_counts.sum(axis=0), axis=1) * 1e6 / 1.0

    # Build labels
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
    """Return features with expression rate > threshold"""
    if df is None:
        return set()
    return set((df > 0).mean(axis=0)[lambda s: s > threshold].index.tolist())

# ============================================================
# Feature selection function (called within each fold)
# ============================================================

def run_bootstrap_lasso_single(X, y, seed):
    X_res, y_res = resample(X, y, random_state=seed, stratify=y)
    model = Lasso(alpha=LASSO_ALPHA, max_iter=5000, random_state=seed)
    model.fit(X_res, y_res)
    return model.coef_ != 0


def select_features_in_fold(X_train_df, y_train, candidate_features):
    """
    Complete feature selection within a fold training partition:
    1) ANOVA Top300 (fit on train only)
    2) StandardScaler.fit (on train only)
    3) Bootstrap LASSO x 100 (on train only)
    Return: selected features list, scaler object
    """
    # Step 1: ANOVA
    X_arr   = np.log1p(X_train_df[candidate_features].values)
    k       = min(ANOVA_K, len(candidate_features))
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_arr, y_train.values)
    top_k_features = np.array(candidate_features)[selector.get_support()]

    # Step 2: StandardScaler (fit on train only)
    X_top = np.log1p(X_train_df[top_k_features].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)

    # Step 3: Bootstrap LASSO (parallel)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_bootstrap_lasso_single)(X_scaled, y_train.values, i)
        for i in range(N_LASSO_BOOT)
    )
    freq = np.array(results).mean(axis=0)

    # Dynamic threshold: try 0.60, fall back to 0.50 if fewer than 3 features
    thr = LASSO_FREQ_THR
    selected_mask = freq >= thr
    if selected_mask.sum() < 3:
        thr = 0.50
        selected_mask = freq >= thr

    selected_features = top_k_features[selected_mask].tolist()
    return selected_features, scaler, top_k_features

# ============================================================
# Model definitions
# ============================================================

if HAS_TORCH:
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=4):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )
        def forward(self, x):
            b, c, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1)
            return x * y.expand_as(x)

    class Attention1DCNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.GroupNorm(1, 32), nn.GELU(), nn.Dropout(0.3),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.GroupNorm(1, 64), nn.GELU(),
                SEBlock(64), nn.AdaptiveMaxPool1d(4), nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4, 32), nn.GELU(), nn.Dropout(0.4), nn.Linear(32, 2)
            )
        def forward(self, x):
            return self.classifier(self.features(x.unsqueeze(1)))

    class SklearnCNN(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim=300, epochs=20, lr=0.001, batch_size=32):
            self.input_dim  = input_dim
            self.epochs     = epochs
            self.lr         = lr
            self.batch_size = batch_size
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, X, y):
            self.model_    = Attention1DCNN(self.input_dim).to(self.device)
            optimizer      = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-3)
            scheduler      = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            criterion      = nn.CrossEntropyLoss()
            loader = DataLoader(
                TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                batch_size=self.batch_size, shuffle=True
            )
            self.model_.train()
            for _ in range(self.epochs):
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    criterion(self.model_(bx), by).backward()
                    optimizer.step()
                scheduler.step()
            return self

        def predict_proba(self, X):
            self.model_.eval()
            with torch.no_grad():
                return torch.softmax(
                    self.model_(torch.FloatTensor(X).to(self.device)), dim=1
                ).cpu().numpy()


class BalancedBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=20):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators

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


def build_model_dict(input_dim_dl):
    """Build all model dictionaries"""
    ml_models = {
        "LR":         LogisticRegression(max_iter=3000, C=0.5, solver='liblinear', class_weight='balanced'),
        "LDA":        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        "NaiveBayes": GaussianNB(),
        "KNN":        KNeighborsClassifier(n_neighbors=51, weights='distance', metric='manhattan'),
        "RF":         RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42),
        "AdaBoost":   AdaBoostClassifier(n_estimators=50, random_state=42),
    }
    if HAS_XGB:
        ml_models["XGB"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            n_jobs=4, tree_method='hist', device='cuda', verbosity=0
        )
    dl_models = {
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                             alpha=0.01, random_state=42),
    }
    if HAS_TORCH:
        dl_models["CNN"] = SklearnCNN(input_dim=input_dim_dl, epochs=40, lr=0.001)
    return ml_models, dl_models

# ============================================================
# Preprocessing (strict leakage prevention)
# ============================================================

def preprocess_split(df_train, df_test_list, method='robust_log1p'):
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values

    if method == 'robust_log1p':
        X_train = np.log1p(np.maximum(X_train, 0))
        scaler  = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
    else:
        scaler  = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())

    X_train_trans = scaler.fit_transform(X_train)
    test_data = []
    for df_test in df_test_list:
        X_test = df_test.drop(columns=['target']).values
        y_test = df_test['target'].values
        if method == 'robust_log1p':
            X_test = np.log1p(np.maximum(X_test, 0))
        test_data.append((scaler.transform(X_test), y_test))
    return (X_train_trans, y_train), test_data

# ============================================================
# ROC curve plotting
# ============================================================

def plot_roc_bootstrap(dataset_name, data_list, filename, n_bootstraps=N_BOOTSTRAPS):
    plt.figure(figsize=(8, 8))
    colors = ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F",
              "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]
    mean_fpr = np.linspace(0, 1, 100)

    for i, (name, y_true, y_prob) in enumerate(data_list):
        tprs, aucs = [], []
        rng = np.random.RandomState(42)
        valid = 0
        while valid < n_bootstraps:
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
            interp_tpr  = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            valid += 1

        mean_tpr      = np.mean(tprs, axis=0)
        mean_tpr[-1]  = 1.0
        plt.plot(mean_fpr, mean_tpr, color=colors[i % len(colors)], lw=1.5,
                 label=f"{name} (AUC={np.mean(aucs):.3f})")

    plt.plot([0,1],[0,1], color='#999999', linestyle='--', lw=1.0)
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.xlabel("1 - Specificity", fontsize=14, fontweight='bold')
    plt.ylabel("Sensitivity",     fontsize=14, fontweight='bold')
    plt.title(f"ROC Curve ({dataset_name})", fontsize=16, fontweight='bold')
    ax = plt.gca()
    for sp in ax.spines.values(): sp.set_linewidth(1.5)
    ax.tick_params(labelsize=12); ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right', frameon=False,
               prop={'weight':'bold','size':10})
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   Saved: {filename}")


def plot_nested_cv_results(nested_results, filename):
    """
    New figure: Nested CV model AUC distribution
    Boxplot with 10-fold AUC, line annotated as mean +/- SD
    """
    model_names = list(nested_results.keys())
    auc_data    = [nested_results[m] for m in model_names]
    means       = [np.mean(d) for d in auc_data]
    sds         = [np.std(d)  for d in auc_data]

    # Sort by mean AUC descending
    order       = np.argsort(means)[::-1]
    model_names = [model_names[i] for i in order]
    auc_data    = [auc_data[i]    for i in order]
    means       = [means[i]       for i in order]
    sds         = [sds[i]         for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(model_names)*1.2), 6))
    colors  = ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F",
               "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]

    bp = ax.boxplot(auc_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate mean +/- SD
    for i, (m, s) in enumerate(zip(means, sds), start=1):
        ax.text(i, m + s + 0.005, f"{m:.3f}±{s:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=0.5, color='grey', linestyle='--', lw=1.0, alpha=0.5)
    ax.set_xticks(range(1, len(model_names)+1))
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax.set_ylabel("AUC (10-fold Nested CV)", fontsize=13, fontweight='bold')
    ax.set_title("Nested Cross-Validation Performance\n"
                 "(Feature selection confined to training folds)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim([max(0, min([min(d) for d in auc_data]) - 0.05), 1.02])
    for sp in ax.spines.values(): sp.set_linewidth(1.5)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   Saved: {filename}")

# ============================================================
# PART A：10-fold Nested Cross-Validation
# ============================================================

def run_nested_cv(tcga_X, tcga_y, candidate_features):
    """
    10-fold Nested CV：
    - Feature selection (ANOVA + LASSO) strictly within each outer fold
    - Return per-model AUC list across 10 folds
    """
    print("\n" + "="*60)
    print("PART A: 10-fold Nested Cross-Validation")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    X_full = tcga_X[candidate_features]
    y_full = tcga_y

    # Initialize container: {model_name: [auc_fold1, ..., auc_fold10]}
    # Use placeholder, init after first fold determines names
    nested_results  = {}
    fold_signatures = []  # Track feature count per fold

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold_idx+1}/{N_OUTER_FOLDS} ---")

        X_train_df = X_full.iloc[train_idx]
        X_test_df  = X_full.iloc[test_idx]
        y_train    = y_full.iloc[train_idx]
        y_test     = y_full.iloc[test_idx]

        # 1) Feature selection (strictly within train partition)
        selected_feats, fold_scaler, top_k_feats = select_features_in_fold(
            X_train_df, y_train, candidate_features
        )
        fold_signatures.append(len(selected_feats))
        print(f"   Feature selection done: ANOVA Top{len(top_k_feats)} -> LASSO -> {len(selected_feats)} features")

        # 2) Prepare ML data (using LASSO-selected features)
        # Re-fit on selected features
        ml_scaler = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
        X_tr_ml   = ml_scaler.fit_transform(
            np.log1p(np.maximum(X_train_df[selected_feats].values, 0))
        )
        X_te_ml   = ml_scaler.transform(
            np.log1p(np.maximum(X_test_df[selected_feats].values, 0))
        )

        # 3) Prepare DL data (using Top300 features)
        dl_feats  = top_k_feats.tolist()
        dl_pipe   = make_pipeline(RobustScaler(quantile_range=(5,95)), Normalizer())
        X_tr_dl   = dl_pipe.fit_transform(
            np.log1p(np.maximum(X_train_df[dl_feats].values, 0))
        )
        X_te_dl   = dl_pipe.transform(
            np.log1p(np.maximum(X_test_df[dl_feats].values, 0))
        )

        y_tr = y_train.values
        y_te = y_test.values

        # 4) Build models, train and evaluate
        ml_models, dl_models = build_model_dict(input_dim_dl=len(dl_feats))

        for m_type, m_group in [("ML", ml_models), ("DL", dl_models)]:
            X_tr = X_tr_ml if m_type == "ML" else X_tr_dl
            X_te = X_te_ml if m_type == "ML" else X_te_dl

            for name, base_model in m_group.items():
                model = BalancedBaggingClassifier(
                    base_model,
                    n_estimators=20 if m_type == "ML" else 10
                )
                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)[:, 1]

                if len(np.unique(y_te)) < 2:
                    print(f"   [SKIP] {name}: test fold has only one class")
                    continue

                fpr, tpr, _ = roc_curve(y_te, y_prob)
                fold_auc    = auc(fpr, tpr)

                if name not in nested_results:
                    nested_results[name] = []
                nested_results[name].append(fold_auc)
                print(f"   {name:12s} AUC = {fold_auc:.4f}")

    # Summary
    print("\n>>> Nested CV summary results:")
    print(f"{'Model':<14} {'Mean AUC':>10} {'SD':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    summary_rows = []
    for name, aucs in nested_results.items():
        m, s = np.mean(aucs), np.std(aucs)
        print(f"{name:<14} {m:>10.4f} {s:>8.4f} {min(aucs):>8.4f} {max(aucs):>8.4f}")
        summary_rows.append({
            "Model": name,
            "Mean_AUC": round(m, 4),
            "SD": round(s, 4),
            "Min_AUC": round(min(aucs), 4),
            "Max_AUC": round(max(aucs), 4),
            "Fold_AUCs": str([round(a, 4) for a in aucs])
        })

    print(f"\n   Features per fold: {fold_signatures} (mean={np.mean(fold_signatures):.1f})")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{OUTPUT_DIR}/nested_cv_summary.csv", index=False)
    print(f"   Saved: {OUTPUT_DIR}/nested_cv_summary.csv")

    return nested_results

# ============================================================
# PART B: Final model (full TCGA train -> external validation)
# ============================================================

def run_final_model(tcga_X, tcga_y, candidate_features,
                    gse1_X, gse1_y, gse2_X, gse2_y):
    """
    Final model: full TCGA feature selection -> training -> external validation
    Logic same as original, only CPM calculation corrected
    """
    print("\n" + "="*60)
    print("PART B: Final model training and external validation")
    print("="*60)

    # 1) Full TCGA feature selection
    print("\n>>> B.1 Full TCGA feature selection...")
    X_full = tcga_X[candidate_features]
    y_full = tcga_y

    X_log = np.log1p(X_full.values)
    k     = min(ANOVA_K, len(candidate_features))
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_log, y_full.values)
    top300_features = np.array(candidate_features)[selector.get_support()].tolist()
    print(f"   ANOVA Top{len(top300_features)} done")

    # Save Supplementary Table: TOP300 Features with ANOVA scores
    top300_scores = selector.scores_[selector.get_support()]
    top300_df = pd.DataFrame({
        'eRNA_ID': top300_features,
        'ANOVA_F_Score': top300_scores
    }).sort_values('ANOVA_F_Score', ascending=False)
    top300_df.to_csv(f"{OUTPUT_DIR}/Supplementary_Table_TOP300_Features.csv", index=False)
    print(f"   Saved: {OUTPUT_DIR}/Supplementary_Table_TOP300_Features.csv")

    # Bootstrap LASSO
    scaler_lasso = StandardScaler()
    X_scaled     = scaler_lasso.fit_transform(np.log1p(X_full[top300_features].values))

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_bootstrap_lasso_single)(X_scaled, y_full.values, i)
        for i in range(N_LASSO_BOOT)
    )
    freq = np.array(results).mean(axis=0)

    thr = LASSO_FREQ_THR
    selected_mask = freq >= thr
    if selected_mask.sum() < 3:
        thr = 0.50
        selected_mask = freq >= thr

    final_signature = np.array(top300_features)[selected_mask].tolist()
    print(f"   Bootstrap LASSO -> {len(final_signature)} final features (threshold={thr})")

    # Save signature
    sig_df = pd.DataFrame({
        'Feature': final_signature,
        'Frequency': freq[selected_mask]
    }).sort_values('Frequency', ascending=False)
    sig_df.to_csv(f"{OUTPUT_DIR}/final_signature_list.csv", index=False)

    # 2) Build datasets
    def make_df(X_df, feats, y_series):
        df = X_df[feats].copy()
        df['target'] = y_series
        return df

    def make_df_top300(X_df, feats, y_series):
        df = X_df[feats].copy()
        df['target'] = y_series
        return df

    # Align external dataset features
    def align_features(X_ext, feats):
        missing = [f for f in feats if f not in X_ext.columns]
        if missing:
            print(f"   [WARN] External data missing {len(missing)} features, filling with 0")
            for f in missing:
                X_ext[f] = 0.0
        return X_ext[feats]

    gse1_X_aligned = align_features(gse1_X.copy(), final_signature)
    gse2_X_aligned = align_features(gse2_X.copy(), final_signature)
    gse1_X_top300  = align_features(gse1_X.copy(), top300_features)
    gse2_X_top300  = align_features(gse2_X.copy(), top300_features)

    df_tcga_ml  = make_df(X_full, final_signature, y_full)
    df_gse1_ml  = make_df(gse1_X_aligned, final_signature, gse1_y)
    df_gse2_ml  = make_df(gse2_X_aligned, final_signature, gse2_y)

    df_tcga_dl  = make_df_top300(X_full, top300_features, y_full)
    df_gse1_dl  = make_df_top300(gse1_X_top300, top300_features, gse1_y)
    df_gse2_dl  = make_df_top300(gse2_X_top300, top300_features, gse2_y)

    # 3) Preprocessing
    print("\n>>> B.2 Preprocessing...")
    (X_tr_ml, y_tr), test_ml = preprocess_split(df_tcga_ml, [df_gse1_ml, df_gse2_ml], 'robust_log1p')
    X_t1_ml, y_t1 = test_ml[0]
    X_t2_ml, y_t2 = test_ml[1]

    (X_tr_dl, y_tr_dl), test_dl = preprocess_split(df_tcga_dl, [df_gse1_dl, df_gse2_dl], 'robust_log1p')
    X_t1_dl, y_t1_dl = test_dl[0]
    X_t2_dl, y_t2_dl = test_dl[1]

    # 4) Training and evaluation
    print(f"\n>>> B.3 Training models with bootstrap evaluation (N={N_BOOTSTRAPS})...")
    ml_models, dl_models = build_model_dict(input_dim_dl=X_tr_dl.shape[1])

    plot_data_train, plot_data_gse1, plot_data_gse2 = [], [], []
    sci_results = []

    for m_type, m_group in [("ML", ml_models), ("DL", dl_models)]:
        X_tr = X_tr_ml if m_type == "ML" else X_tr_dl
        y_tr_use = y_tr if m_type == "ML" else y_tr_dl
        X_t1 = X_t1_ml if m_type == "ML" else X_t1_dl
        X_t2 = X_t2_ml if m_type == "ML" else X_t2_dl
        y_t1_use = y_t1 if m_type == "ML" else y_t1_dl
        y_t2_use = y_t2 if m_type == "ML" else y_t2_dl

        for name, base_model in m_group.items():
            print(f"   {name}...", end="", flush=True)
            model = BalancedBaggingClassifier(
                base_model, n_estimators=20 if m_type == "ML" else 10
            )
            model.fit(X_tr, y_tr_use)

            # Save model
            try:
                joblib.dump(model, f"{OUTPUT_DIR}/{name}_{m_type}_model.pkl")
            except Exception as e:
                print(f"\n   [WARN] {name} save failed: {e}")

            p_tr = model.predict_proba(X_tr)[:, 1]
            p_t1 = model.predict_proba(X_t1)[:, 1]
            p_t2 = model.predict_proba(X_t2)[:, 1]

            plot_data_train.append((name, y_tr_use, p_tr))
            plot_data_gse1.append((name, y_t1_use, p_t1))
            plot_data_gse2.append((name, y_t2_use, p_t2))

            res_t1, auc_t1 = evaluate_with_bootstrap(y_t1_use, p_t1)
            res_t2, auc_t2 = evaluate_with_bootstrap(y_t2_use, p_t2)
            print(f" GSE225846 AUC={auc_t1:.3f} | GSE229571 AUC={auc_t2:.3f}")

            for ds_name, res in zip(["GSE225846", "GSE229571"], [res_t1, res_t2]):
                sci_results.append({
                    "Dataset": ds_name, "Model": name, "Type": m_type,
                    "AUC (95% CI)":         res[0],
                    "Accuracy (95% CI)":    res[1],
                    "Sensitivity (95% CI)": res[2],
                    "Specificity (95% CI)": res[3],
                    "Precision (95% CI)":   res[4],
                    "F1_Score (95% CI)":    res[5],
                })

    # 5) Save results
    print("\n>>> B.4 Saving results...")
    df_sci = pd.DataFrame(sci_results)
    df_sci.to_csv(f"{OUTPUT_DIR}/SCI_metrics_table_external_validation.csv", index=False)
    print(f"   Saved: {OUTPUT_DIR}/SCI_metrics_table_external_validation.csv")

    plot_roc_bootstrap("GSE225846 Validation", plot_data_gse1,
                       f"{OUTPUT_DIR}/ROC_GSE225846_Smoothed.svg")
    plot_roc_bootstrap("GSE229571 Validation", plot_data_gse2,
                       f"{OUTPUT_DIR}/ROC_GSE229571_Smoothed.svg")

    # Save DL scaler for Shiny
    try:
        dl_scaler = make_pipeline(RobustScaler(quantile_range=(5,95)), Normalizer())
        X_raw = np.log1p(np.maximum(X_full[top300_features].values, 0))
        dl_scaler.fit(X_raw)
        joblib.dump(dl_scaler, f"{OUTPUT_DIR}/DL_robust_log1p_scaler.pkl")
        print(f"   Saved: {OUTPUT_DIR}/DL_robust_log1p_scaler.pkl")
    except Exception as e:
        print(f"   [WARN] Scaler save failed: {e}")

    return nested_results if 'nested_results' in dir() else None

# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":
    seed_everything(42)
    print("="*60)
    print("Suggestion 1.1 - Nested CV Pipeline")
    print("="*60)

    # -- STEP 0: Load data ----------------------------------------
    print("\n>>> STEP 0: Loading data with global technical feasibility filter")
    tcga_X, tcga_y = load_tcga_data()
    gse1_X, gse1_y = load_external_data(FILES["GSE225846"])
    gse2_X, gse2_y = load_external_data(FILES["GSE229571"])

    if gse1_X is None or gse2_X is None:
        print("ERROR: External data loading failed")
        sys.exit(1)

    # Three-dataset intersection (technical feasibility, label-free)
    print("\n>>> Three-dataset intersection filtering (label-free feasibility)...")
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

    # ── PART A：Nested CV ────────────────────────────────────
    nested_results = run_nested_cv(tcga_X, tcga_y, candidate_features)

    # Plot Nested CV supplementary figure
    plot_nested_cv_results(
        nested_results,
        f"{OUTPUT_DIR}/FigS_NestedCV_AUC_Distribution.svg"
    )

    # -- PART B: Final model ----------------------------------------
    run_final_model(tcga_X, tcga_y, candidate_features,
                    gse1_X, gse1_y, gse2_X, gse2_y)

    print("\n" + "="*60)
    print("All done! Results saved to:", OUTPUT_DIR)
    print("="*60)
