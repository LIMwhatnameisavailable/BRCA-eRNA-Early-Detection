import pandas as pd
import numpy as np
import os
import sys
import random

# ==================== 路径配置 ====================

BASE_DIR = "../Data_Source"
OUTPUT_DIR = "../results/clean_data_early_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLINICAL_FILE = "/mnt/disk2/srtp2024/TJW/TCGA-BRCA.clinical.xlsx"

FILES = {
    "TCGA_Expr": os.path.join(BASE_DIR, "TCGA/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt"),
    "TCGA_Normal_List": os.path.join(BASE_DIR, "TCGA/Normal_like_samples.csv"),

    "GSE1": {
        "name": "GSE225846",
        "expr": os.path.join(BASE_DIR, "GSE225846/analysis/counts_matrix_500bp_clean.txt"),
        "meta": os.path.join(BASE_DIR, "GSE225846/SraRunTable_GSE225846.csv"),
        "stage_col": "stage",
        "type_col": "type"
    },

    "GSE2": {
        "name": "GSE229571",
        "expr": os.path.join(BASE_DIR, "GSE229571/analysis/counts_matrix_500bp_clean.txt"),
        "meta": os.path.join(BASE_DIR, "GSE229571/SraRunTable_GSE229571.csv"),
        "stage_col": "tumor_stage",
        "type_col": "tissue"
    }
}

# 早期分期标签（I/II 期及其亚型）
EARLY_STAGES = [
    'Stage I', 'Stage IA', 'Stage IB',
    'Stage II', 'Stage IIA', 'Stage IIB',
    '1', '2', 'I', 'II'
]

# ==================== 通用函数 ====================

def seed_everything(seed=42):
    """固定所有环境的随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_selected_features():
    """读取 Step 1 生成的最终特征列表"""
    signature_file = os.path.join(OUTPUT_DIR, "final_signature_list.csv")
    if not os.path.exists(signature_file):
        print(f"错误: 找不到特征文件 {signature_file}。请先运行 Step 1。")
        sys.exit(1)

    df_sig = pd.read_csv(signature_file)
    features = df_sig['Feature'].tolist()
    print(f"成功: 已从 Step 1 结果中读取 {len(features)} 个特征。")
    return features


def process_tcga(selected_features):
    print(f"\n>>> 处理 TCGA (Strict Early Stage)...")

    try:
        pheno = pd.read_csv(CLINICAL_FILE, sep='\t', encoding='utf-16')
    except:
        pheno = pd.read_csv(CLINICAL_FILE, sep='\t')

    stage_col = 'ajcc_pathologic_stage.diagnoses'
    if stage_col not in pheno.columns:
        cols = [c for c in pheno.columns if 'stage' in c.lower() and 'diagnoses' in c.lower()]
        if cols:
            stage_col = cols[0]

    mask = pheno[stage_col].isin(EARLY_STAGES)
    early_patients = set(pheno.loc[mask, 'submitter_id'].values)
    print(f"   TCGA Early Stage 病人数量: {len(early_patients)}")

    print("   读取表达矩阵...")
    df = pd.read_csv(FILES["TCGA_Expr"], sep='\t', index_col=0)

    normal_df = pd.read_csv(FILES["TCGA_Normal_List"])
    target_col = [c for c in normal_df.columns if normal_df[c].astype(str).str.contains('TCGA').any()][0]
    whitelist_normals = set([f"{x[:12]}_normal" for x in normal_df[target_col]])

    final_samples = []

    for col in df.columns:
        if col in whitelist_normals:
            final_samples.append((col, 0))
        elif '_tumor' in col:
            pid = col[:12]
            if pid in early_patients:
                final_samples.append((col, 1))

    sample_df = pd.DataFrame(final_samples, columns=['sample_id', 'target'])
    print(f"   最终纳入: Normal={sum(sample_df['target']==0)}, Tumor={sum(sample_df['target']==1)}")

    try:
        valid_ids = sample_df['sample_id'].tolist()
        final_df = df.loc[selected_features, valid_ids].T

        final_df['target'] = sample_df.set_index('sample_id')['target']

        save_path = f"{OUTPUT_DIR}/dataset_tcga.csv"
        final_df.to_csv(save_path)
        print(f"   成功: 已保存 {save_path}")

    except KeyError as e:
        print(f"   错误: 特征提取失败，某些特征在矩阵中不存在: {e}")


def process_gse(config, selected_features):
    name = config['name']
    print(f"\n>>> 处理 {name} (Strict Early Stage)...")

    if not os.path.exists(config['meta']) or not os.path.exists(config['expr']):
        print(f"   跳过: 文件不存在 ({name})")
        return

    meta = pd.read_csv(config['meta'])

    print(f"   读取表达矩阵: {config['expr']}")
    try:
        expr = pd.read_csv(config['expr'], sep='\t', comment='#', index_col=0)
        expr.columns = [c.replace('.bam', '').strip() for c in expr.columns]

        drop_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
        expr = expr.drop(columns=[c for c in drop_cols if c in expr.columns])

        # 将 counts 转换为 CPM（窗口长度 500bp，除以 0.5kb）
        expr = expr.div(expr.sum(axis=0), axis=1) * 1e6 / 0.5

    except Exception as e:
        print(f"   错误: 读取矩阵失败: {e}")
        return

    id_col = 'Run' if 'Run' in meta.columns else meta.columns[0]
    type_col = config['type_col']
    stage_col = config['stage_col']

    valid_samples = []
    labels = []

    for _, row in meta.iterrows():
        sample_id = row[id_col]

        if sample_id not in expr.columns:
            continue

        tissue = str(row[type_col]).lower()

        if stage_col in row and pd.notna(row[stage_col]):
            stage = str(row[stage_col]).replace("STAGE", "").strip()
        else:
            stage = ""

        is_normal = 'normal' in tissue or 'healthy' in tissue
        is_tumor = not is_normal

        if is_normal:
            valid_samples.append(sample_id)
            labels.append(0)
        elif is_tumor:
            s_up = stage.upper()
            if any(x in s_up for x in ['I', '1', '2']) and not any(x in s_up for x in ['III', 'IV', '3', '4']):
                valid_samples.append(sample_id)
                labels.append(1)

    print(f"   [{name}] 筛选后样本: {len(valid_samples)} (Normal={labels.count(0)}, Tumor={labels.count(1)})")

    if len(valid_samples) == 0:
        print("   警告: 没有样本被选中，请检查 metadata 的 stage 写法。")
        return

    try:
        final_df = expr.loc[selected_features, valid_samples].T
        final_df['target'] = labels

        save_path = f"{OUTPUT_DIR}/dataset_{name.lower()}.csv"
        final_df.to_csv(save_path)
        print(f"   成功: 已保存 {save_path}")
    except KeyError as e:
        print(f"   错误: {name} 特征提取失败，可能是特征名不匹配: {e}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    seed_everything(42)

    # 加载 Step 1 筛选出的特征
    dynamic_features = load_selected_features()

    process_tcga(dynamic_features)
    process_gse(FILES["GSE1"], dynamic_features)
    process_gse(FILES["GSE2"], dynamic_features)

    print("\n>>> 数据准备完成！")
