import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.pipeline import make_pipeline

# ================= Configuration =================
DATA_DIR = "./clean_data_early_stage"
OUTPUT_DIR = "./results_batch_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS_TYPE = {"Normal": "#457B9D", "Tumor": "#E63946"}
COLORS_BATCH = {"TCGA": "#2A9D8F", "GSE225846": "#E9C46A", "GSE229571": "#F4A261"}

# ================= 1. Load data =================
df_tcga = pd.read_csv(f"{DATA_DIR}/dataset_tcga.csv", index_col=0)
df_gse1 = pd.read_csv(f"{DATA_DIR}/dataset_gse225846.csv", index_col=0)
df_gse2 = pd.read_csv(f"{DATA_DIR}/dataset_gse229571.csv", index_col=0)

X_tcga = df_tcga.drop(columns=['target']).values
X_gse1 = df_gse1.drop(columns=['target']).values
X_gse2 = df_gse2.drop(columns=['target']).values

batch_labels = ['TCGA']*len(df_tcga) + ['GSE225846']*len(df_gse1) + ['GSE229571']*len(df_gse2)
bio_labels = (['Tumor' if y == 1 else 'Normal' for y in df_tcga['target']] +
              ['Tumor' if y == 1 else 'Normal' for y in df_gse1['target']] +
              ['Tumor' if y == 1 else 'Normal' for y in df_gse2['target']])
df_meta = pd.DataFrame({'Batch': batch_labels, 'Type': bio_labels})

# ================= 2. Execute final method only =================
# 1. Baseline Log1p
X_tcga_log = np.log1p(X_tcga)
X_gse1_log = np.log1p(X_gse1)
X_gse2_log = np.log1p(X_gse2)

# 2. Leak-proof Robust + L2 (fit only on TCGA)
pipeline = make_pipeline(RobustScaler(), Normalizer())
pipeline.fit(X_tcga_log)

X_tcga_trans = pipeline.transform(X_tcga_log)
X_gse1_trans = pipeline.transform(X_gse1_log)
X_gse2_trans = pipeline.transform(X_gse2_log)

# 3. Concatenate and PCA 
X_concat_trans = np.vstack([X_tcga_trans, X_gse1_trans, X_gse2_trans])
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_concat_trans)
var_explained = pca.explained_variance_ratio_ * 100

df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
df_pca['Batch'] = df_meta['Batch']
df_pca['Type'] = df_meta['Type']

# ================= 3. Plot (same style as original figures) =================
def apply_theme_classic(ax, var1, var2):
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", color='black', fontsize=14)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", color='black', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', colors='black', labelsize=12, width=1)
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), 
                  ncol=len(labels), frameon=False, fontsize=12, handletextpad=0.1)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left panel: by batch
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Batch', palette=COLORS_BATCH, 
                ax=axes[0], s=80, edgecolor='white', linewidth=0.5, alpha=0.8)
apply_theme_classic(axes[0], var_explained[0], var_explained[1])

# Right panel: by biological type
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Type', palette=COLORS_TYPE, 
                ax=axes[1], s=80, edgecolor='white', linewidth=0.5, alpha=0.8)
apply_theme_classic(axes[1], var_explained[0], var_explained[1])

plt.tight_layout()

# Save replacement figure
plt.savefig(f"{OUTPUT_DIR}/Fig_PCA_Replacement_BottomRight.svg", format='svg', bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/Fig_PCA_Replacement_BottomRight.png", format='png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 替换用的右下角子图已生成！请前往文件夹提取。")
