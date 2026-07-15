import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, Normalizer
from sklearn.pipeline import make_pipeline

# ================= Configuration =================
DATA_DIR = "./clean_data_early_stage"
OUTPUT_DIR = "./results_batch_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme matching paper aesthetic
COLORS_TYPE = {"Normal": "#457B9D", "Tumor": "#E63946"}
# Batch colors (matching R classic style)
COLORS_BATCH = {"TCGA": "#2A9D8F", "GSE225846": "#E9C46A", "GSE229571": "#F4A261"}

# ================= 1. Load data =================
print(">>> 加载数据...")
try:
    df_tcga = pd.read_csv(f"{DATA_DIR}/dataset_tcga.csv", index_col=0)
    df_gse1 = pd.read_csv(f"{DATA_DIR}/dataset_gse225846.csv", index_col=0)
    df_gse2 = pd.read_csv(f"{DATA_DIR}/dataset_gse229571.csv", index_col=0)
except FileNotFoundError:
    print("❌ 找不到特征数据文件，请确认路径。")
    exit()

# Merge all datasets
datasets = {"TCGA": df_tcga, "GSE225846": df_gse1, "GSE229571": df_gse2}
X_all = []
batch_labels = []
bio_labels = []

for name, df in datasets.items():
    X_all.append(df.drop(columns=['target']))
    batch_labels.extend([name] * len(df))
    bio_labels.extend(['Tumor' if y == 1 else 'Normal' for y in df['target']])

X_concat = pd.concat(X_all).values
df_meta = pd.DataFrame({'Batch': batch_labels, 'Type': bio_labels})

# ================= 2. Define methods to compare =================
# Includes baseline methods plus selected ML/DL core methods
methods = {
    "Baseline: Log1p (No Correction)": None, 
    "Baseline: Z-Score (StandardScaler)": StandardScaler(),
    "Proposed ML: Robust + L2 Norm": make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer()),
    "Proposed DL: RankGauss": QuantileTransformer(output_distribution='normal', random_state=42)
}

# ================= 3. R-style plotting function (title removed) =================
def apply_theme_classic(ax, var1, var2):
    """Replicate R theme_classic() with specified matplotlib settings, no title"""
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", color='black', fontsize=14)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", color='black', fontsize=14)
    
    # Hide top and right spines (theme_classic)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Emphasize left and bottom spines
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    
    # Axis ticks and background
    ax.tick_params(axis='both', colors='black', labelsize=12, width=1)
    ax.set_facecolor('white')
    ax.grid(False)
    
    # Force square aspect ratio (aspect.ratio = 1)
    ax.set_box_aspect(1)
    
    # Top legend (legend.position = "top", legend.title = element_blank())
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), 
                  ncol=len(labels), frameon=False, fontsize=12, handletextpad=0.1)

# ================= 4. Execute transforms and plot =================
for method_name, transformer in methods.items():
    print(f"\n>>> 处理方法: {method_name}")
    
    # Baseline Log1p transformation
    X_trans = np.log1p(X_concat)
    
    # Additional method transformations
    if transformer is not None:
        X_trans = transformer.fit_transform(X_trans)
        
    # PCA dimensionality reduction
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_trans)
    var_explained = pca.explained_variance_ratio_ * 100
    
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    df_pca['Batch'] = df_meta['Batch']
    df_pca['Type'] = df_meta['Type']
    
    # --- Plot layout (1 row, 2 cols: left=by batch, right=by biological type) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Panel 1: Color by Batch (check batch effect removal)
    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue='Batch',
        palette=COLORS_BATCH, ax=axes[0],
        s=80, edgecolor='white', linewidth=0.5, alpha=0.8
    )
    apply_theme_classic(axes[0], var_explained[0], var_explained[1])
    
    # Panel 2: Color by Type (check biological signal preservation)
    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue='Type',
        palette=COLORS_TYPE, ax=axes[1],
        s=80, edgecolor='white', linewidth=0.5, alpha=0.8
    )
    apply_theme_classic(axes[1], var_explained[0], var_explained[1])
    
    plt.tight_layout()
    
    # Save results for this method (SVG and PNG)
    safe_name = method_name.replace(":", "").replace(" ", "_").replace("/", "_").replace("+", "and")
    svg_path = f"{OUTPUT_DIR}/Fig_PCA_{safe_name}.svg"
    png_path = f"{OUTPUT_DIR}/Fig_PCA_{safe_name}.png"
    
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 已保存: {svg_path}")
    print(f"   ✅ 已保存: {png_path}")

print("\n🎉 全部方法评估完成！所有图片已保存至独立文件夹。")