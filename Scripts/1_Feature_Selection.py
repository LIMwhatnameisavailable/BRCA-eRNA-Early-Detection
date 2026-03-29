import pandas as pd
import numpy as np
import os
import sys
import warnings
import random
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================

BASE_DIR = "/mnt/disk4/srtp2024"
OUTPUT_DIR = "./clean_data_early_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 临床数据路径
CLINICAL_PATH = "/mnt/disk2/srtp2024/TJW/TCGA-BRCA.clinical.xlsx"

FILES = {
    "TCGA_Expr": os.path.join(BASE_DIR, "TCGA/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt"),
    "TCGA_Normal_List": os.path.join(BASE_DIR, "TCGA/Normal_like_samples.csv"),

    "GSE225846": {
        "name": "GSE225846",
        "expr": os.path.join(BASE_DIR, "GSE225846/analysis/counts_matrix_500bp_clean.txt"),
        "meta": os.path.join(BASE_DIR, "GSE225846/SraRunTable_GSE225846.csv"),
        "stage_col": "stage",
        "type_col": "type"
    },

    "GSE229571": {
        "name": "GSE229571",
        "expr": os.path.join(BASE_DIR, "GSE229571/analysis/counts_matrix_500bp_clean.txt"),
        "meta": os.path.join(BASE_DIR, "GSE229571/SraRunTable_GSE229571.csv"),
        "stage_col": "tumor_stage",
        "type_col": "tissue"
    }
}

# 目标分期：早期（I/II 期）
TARGET_STAGES = [
    'Stage I', 'Stage IA', 'Stage IB',
    'Stage II', 'Stage IIA', 'Stage IIB'
]

# ==================== 核心函数 ====================

def seed_everything(seed=42):
    """固定所有环境的随机种子，确保结果可复现"""
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


def load_tcga_data_strict():
    print(f"\n>>> 1. 加载并清洗 TCGA 数据...")

    if not os.path.exists(FILES["TCGA_Expr"]):
        print(f"错误: 找不到文件 {FILES['TCGA_Expr']}")
        sys.exit(1)
    df = pd.read_csv(FILES["TCGA_Expr"], sep='\t', index_col=0)
    print("主数据前5个列名:", df.columns[:5].tolist())

    if not os.path.exists(FILES["TCGA_Normal_List"]):
        print(f"错误: 找不到 Normal List")
        sys.exit(1)
    normal_df = pd.read_csv(FILES["TCGA_Normal_List"], header=None, names=['SampleID'])
    target_normal_ids = set([f"{str(x).strip()[:12]}_normal" for x in normal_df['SampleID']])
    final_normals = [col for col in df.columns if col in target_normal_ids]
    print(f"Normal List 原始数量: {len(normal_df)}")
    print(f"主数据中成功匹配到的 Normal 样本数: {len(final_normals)}")

    try:
        pheno = pd.read_csv(CLINICAL_PATH, sep='\t', encoding='utf-16')

        stage_col = 'ajcc_pathologic_stage.diagnoses'
        if stage_col not in pheno.columns:
            stage_col = [c for c in pheno.columns if 'stage' in c.lower() and 'diagnoses' in c.lower()][0]

        patient_col = 'submitter_id'

        mask = pheno[stage_col].isin(TARGET_STAGES)
        early_patients = set(pheno.loc[mask, patient_col].values)
        print(f"   TCGA Early Stage 病人总数: {len(early_patients)}")

    except Exception as e:
        print(f"错误: 临床数据读取失败: {e}")
        sys.exit(1)

    valid_tumor = []
    for col in df.columns:
        if '_tumor' in col.lower():
            patient_id = col[:12]
            if patient_id in early_patients:
                valid_tumor.append(col)

    print(f"   [TCGA] Normal: {len(final_normals)}, Tumor: {len(valid_tumor)}")

    df_final = df[final_normals + valid_tumor].T
    labels = pd.Series(0, index=df_final.index)
    labels[valid_tumor] = 1

    return df_final, labels


def load_external_data(config):
    name = config['name']
    print(f"\n>>> 加载外部数据 {name}...")

    if not os.path.exists(config['meta']) or not os.path.exists(config['expr']):
        print(f"错误: 文件缺失: {name}")
        return None

    meta = pd.read_csv(config['meta'])
    valid_ids = []
    id_col = 'Run' if 'Run' in meta.columns else meta.columns[0]

    for _, row in meta.iterrows():
        cls = get_sample_class(row, config['type_col'], config['stage_col'])
        if cls in ["Normal", "Tumor"]:
            valid_ids.append(row[id_col])

    print(f"   读取矩阵: {config['expr']}")
    try:
        df = pd.read_csv(config['expr'], sep='\t', comment='#', index_col=0)
        df.columns = [c.replace('.bam', '').strip() for c in df.columns]

        drop_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        common_ids = [x for x in valid_ids if x in df.columns]
        print(f"   [{name}] 匹配成功样本数: {len(common_ids)}")

        if len(common_ids) == 0:
            print(f"   警告: 依然没有匹配到样本。")
            print(f"   Metadata ID 示例: {valid_ids[:3]}")
            print(f"   Matrix Col 示例: {df.columns[:3].tolist()}")
            return None

        # 将 counts 转换为 CPM（窗口长度 500bp，除以 0.5kb）
        df_counts = df[common_ids]
        df_cpm = df_counts.div(df_counts.sum(axis=0), axis=1) * 1e6 / 0.5
        return df_cpm.T

    except Exception as e:
        print(f"错误: 读取矩阵失败: {e}")
        return None


def get_highly_expressed_features(df, name, threshold):
    if df is None:
        return set()
    ratio = (df > 0).mean(axis=0)
    selected = ratio[ratio > threshold].index.tolist()
    return set(selected)


# ==================== 主程序 ====================

if __name__ == "__main__":
    seed_everything(42)
    print(">>> 开始 Step 1: 特征筛选流程")

    tcga_X, tcga_y = load_tcga_data_strict()
    gse225846_X = load_external_data(FILES["GSE225846"])
    gse229571_X = load_external_data(FILES["GSE229571"])

    if gse225846_X is None or gse229571_X is None:
        print("错误: 外部数据加载失败，无法继续。")
        sys.exit(1)

    print("\n>>> 2. 三数据集交集过滤低表达特征...")
    final_features = []

    for thresh in [0.5, 0.3, 0.2, 0.1]:
        print(f"   尝试表达率阈值 > {thresh*100}%...")
        f_tcga = get_highly_expressed_features(tcga_X, "TCGA", thresh)
        f_gse1 = get_highly_expressed_features(gse225846_X, "GSE225846", thresh)
        f_gse2 = get_highly_expressed_features(gse229571_X, "GSE229571", thresh)

        common = list(f_tcga & f_gse1 & f_gse2)
        print(f"   -> 三个数据集交集基因数: {len(common)}")

        if len(common) >= 500:
            final_features = common
            break

    if not final_features:
        print("错误：找不到足够的共同基因。")
        sys.exit(1)

    print("\n>>> 3. ANOVA 初筛 (Top 300)...")
    X_train = tcga_X[final_features]
    y_train = tcga_y

    X_train_log = np.log1p(X_train)

    K_DL = 300
    selector = SelectKBest(f_classif, k=min(K_DL, X_train_log.shape[1]))
    X_train_selected = selector.fit_transform(X_train_log, y_train)

    top300_features = np.array(final_features)[selector.get_support()]
    print(f"   已选出 Top {len(top300_features)} 基因。")

    all_scores = selector.scores_
    support = selector.get_support()

    top300_df = pd.DataFrame({
        'eRNA_ID': np.array(final_features)[support],
        'ANOVA_F_Score': all_scores[support]
    })
    top300_df = top300_df.sort_values(by='ANOVA_F_Score', ascending=False)

    top300_save_path = os.path.join(OUTPUT_DIR, "Supplementary_Table_TOP300_Features.csv")
    top300_df.to_csv(top300_save_path, index=False)
    print(f"   成功: 已保存 TOP300 特征及 F-score 附表至: {top300_save_path}")

    def save_dataset(df_X, features, output_name, meta_config=None):
        df_sub = df_X[features].copy()

        if meta_config:
            # 外部数据集：从 metadata 中获取样本标签
            meta = pd.read_csv(meta_config['meta'])
            targets = []
            for idx in df_sub.index:
                row = meta[meta.iloc[:, 0] == idx]
                if len(row) > 0:
                    cls = get_sample_class(row.iloc[0], meta_config['type_col'], meta_config['stage_col'])
                    targets.append(1 if cls == "Tumor" else 0)
                else:
                    targets.append(np.nan)
            df_sub['target'] = targets
            df_sub = df_sub.dropna(subset=['target'])
        else:
            # TCGA 数据集：直接使用预生成标签
            df_sub['target'] = tcga_y

        save_p = f"{OUTPUT_DIR}/{output_name}"
        df_sub.to_csv(save_p)
        print(f"      -> {save_p}")

    print(f"   保存 Top 300 数据集...")
    save_dataset(tcga_X, top300_features, "dataset_tcga_top300.csv")
    save_dataset(gse225846_X, top300_features, "dataset_gse225846_top300.csv", FILES["GSE225846"])
    save_dataset(gse229571_X, top300_features, "dataset_gse229571_top300.csv", FILES["GSE229571"])

    print(f"\n>>> 4. 启动 Lasso 稳定性选择...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)

    def run_bootstrap_lasso(X, y, features, seed):
        X_res, y_res = resample(X, y, random_state=seed, stratify=y)
        model = Lasso(alpha=0.015, max_iter=5000, random_state=seed)
        model.fit(X_res, y_res)
        return model.coef_ != 0

    n_iterations = 100
    results = Parallel(n_jobs=-1)(
        delayed(run_bootstrap_lasso)(X_train_scaled, y_train, top300_features, i)
        for i in range(n_iterations)
    )

    results_matrix = np.array(results)
    frequencies = results_matrix.mean(axis=0)

    stability_df = pd.DataFrame({
        'Feature': top300_features,
        'Frequency': frequencies
    }).sort_values(by='Frequency', ascending=False)

    THRESHOLD = 0.60
    final_signature = stability_df[stability_df['Frequency'] >= THRESHOLD]

    print(f"\n>>> 最终筛选结果 (频率 >= {THRESHOLD*100}%):")
    print(f"   共选中 {len(final_signature)} 个特征")
    print(final_signature)

    if len(final_signature) < 3:
        print("警告: 特征过少，自动降低阈值到 0.50...")
        final_signature = stability_df[stability_df['Frequency'] >= 0.50]
        print(f"   新选中 {len(final_signature)} 个特征")

    final_genes = final_signature['Feature'].values

    print(f"\n>>> 5. 保存最终精选数据集 ({len(final_genes)} features)...")

    save_dataset(tcga_X, final_genes, "dataset_tcga_final.csv")
    save_dataset(gse225846_X, final_genes, "dataset_gse225846_final.csv", FILES["GSE225846"])
    save_dataset(gse229571_X, final_genes, "dataset_gse229571_final.csv", FILES["GSE229571"])

    final_signature.to_csv(f"{OUTPUT_DIR}/final_signature_list.csv", index=False)

    print("\n>>> Step 1 完成！")
