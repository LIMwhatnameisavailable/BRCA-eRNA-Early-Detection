# 优化版 step3: 引入 Log1p-Robust 预处理与 Attention-CNN，严格遵循临床防泄露原则
# 不再使用批次效应处理
# 加入shiny框架接口，保存pkl文件
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix, precision_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler, QuantileTransformer, Normalizer, StandardScaler
from sklearn.pipeline import make_pipeline

# ================= 新增：引入模型保存库 =================
import joblib
# ========================================================

# ================= 模型库 =================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ================= PyTorch 环境 =================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ================= 配置 =================
DATA_DIR = "./clean_data_early_stage"
OUTPUT_DIR = "./results_step3_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_BOOTSTRAPS = 5000

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ================= 预处理 (严谨防泄露版) =================
def preprocess_split(df_train, df_test_list, method='robust_log1p'):
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    
    # 核心升级：复刻 0324 中表现最好的 Log1p + Global Robust
    if method == 'robust_log1p':
        X_train = np.log1p(np.maximum(X_train, 0)) # 防止负数报错
        scaler = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
    elif method == 'robust':
        scaler = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
    elif method == 'rank':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    
    # 严格：仅在 Train 上 fit
    X_train_trans = scaler.fit_transform(X_train)
    
    test_data = []
    for df_test in df_test_list:
        X_test = df_test.drop(columns=['target']).values
        y_test = df_test['target'].values
        
        if method == 'robust_log1p':
            X_test = np.log1p(np.maximum(X_test, 0))
            
        # 严格：在 Test 上仅 transform
        X_test_trans = scaler.transform(X_test)
        test_data.append((X_test_trans, y_test))
        
    return (X_train_trans, y_train), test_data

# ================= DL 模型与 Bagging =================
if HAS_TORCH:
    class SEBlock(nn.Module):
        """通道注意力机制：动态赋予重要基因特征更高的权重"""
        def __init__(self, channel, reduction=4):
            super(SEBlock, self).__init__()
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
        """升级版 1D-CNN：GroupNorm + GELU + SE-Attention"""
        def __init__(self, input_dim):
            super(Attention1DCNN, self).__init__()
            # GroupNorm(1, C) 等效于不受 Batch Size 影响的 LayerNorm，非常适合临床单样本预测
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2), 
                nn.GroupNorm(1, 32), 
                nn.GELU(), 
                nn.Dropout(0.3),
                
                nn.Conv1d(32, 64, kernel_size=3, padding=1), 
                nn.GroupNorm(1, 64), 
                nn.GELU(),
                SEBlock(64), # 引入注意力机制
                nn.AdaptiveMaxPool1d(4), 
                nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4, 32), 
                nn.GELU(), 
                nn.Dropout(0.4), 
                nn.Linear(32, 2)
            )
            
        def forward(self, x):
            return self.classifier(self.features(x.unsqueeze(1)))

    class SklearnCNN(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim=300, epochs=40, lr=0.001, batch_size=32):
            self.input_dim = input_dim; self.epochs = epochs; self.lr = lr; self.batch_size = batch_size
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        def fit(self, X, y):
            self.model = Attention1DCNN(self.input_dim).to(self.device)
            # 使用更先进的 AdamW 和余弦退火学习率
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            criterion = nn.CrossEntropyLoss()
            
            loader = DataLoader(TensorDataset(torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device)), 
                                batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for _ in range(self.epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(self.model(bx), by).backward()
                    optimizer.step()
                scheduler.step()
            return self
            
        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                return torch.softmax(self.model(torch.FloatTensor(X).to(self.device)), dim=1).cpu().numpy()

class BalancedBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=20):
        self.base_estimator = base_estimator; self.n_estimators = n_estimators
    def fit(self, X, y):
        self.estimators_ = []
        X_neg, X_pos = X[y == 0], X[y == 1]
        n_pos_sample = max(1, len(X_neg))
        for i in range(self.n_estimators):
            X_balanced = np.vstack((X_neg, resample(X_pos, n_samples=n_pos_sample, random_state=i)))
            y_balanced = np.hstack((np.zeros(len(X_neg)), np.ones(n_pos_sample)))
            self.estimators_.append(clone(self.base_estimator).fit(X_balanced, y_balanced))
        return self
    def predict_proba(self, X):
        return np.mean([clf.predict_proba(X) for clf in self.estimators_], axis=0)

# ================= 核心：Bootstrap 评估引擎 =================
def compute_single_metrics(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= best_threshold).astype(int)
    
    auc_val = auc(fpr, tpr)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return [auc_val, acc, sens, spec, prec, f1]

def evaluate_with_bootstrap(y_true, y_prob, n_bootstraps=N_BOOTSTRAPS, seed=42):
    np.random.seed(seed)
    metrics_list = []
    valid_iters = 0
    
    while valid_iters < n_bootstraps:
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_boot = y_true[indices]
        prob_boot = y_prob[indices]
        
        if len(np.unique(y_boot)) < 2:
            continue
            
        metrics_list.append(compute_single_metrics(y_boot, prob_boot))
        valid_iters += 1
        
    metrics_arr = np.array(metrics_list)
    means = np.mean(metrics_arr, axis=0)
    lower = np.percentile(metrics_arr, 2.5, axis=0)
    upper = np.percentile(metrics_arr, 97.5, axis=0)
    
    formatted = [f"{m:.3f} ({l:.3f}-{u:.3f})" for m, l, u in zip(means, lower, upper)]
    return formatted, means[0]

# ================= 平滑版 ROC 绘制 =================
def plot_roc_bootstrap_R(dataset_name, data_list, filename, n_bootstraps=N_BOOTSTRAPS):
    plt.figure(figsize=(8, 8)) 
    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", 
              "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]
    
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (name, y_true, y_prob) in enumerate(data_list):
        tprs = []
        aucs = []
        np.random.seed(42)
        
        valid_iters = 0
        while valid_iters < n_bootstraps:
            indices = np.random.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2: continue
            
            fpr, tpr, _ = roc_curve(y_true[indices], y_prob[indices])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            valid_iters += 1
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color=colors[i % len(colors)], lw=1.5, 
                 label=f"{name} (AUC = {mean_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='#999999', linestyle='--', lw=1.0)
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=14, fontweight='bold')
    plt.ylabel("Sensitivity (True Positive Rate)", fontsize=14, fontweight='bold')
    plt.title(f"ROC Curve ({dataset_name} - Bootstrap Smoothed)", fontsize=16, fontweight='bold', pad=15)

    ax = plt.gca()
    for spine in ax.spines.values(): spine.set_linewidth(1.5) 
    ax.tick_params(labelsize=12, direction='out')
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), frameon=False, prop={'weight': 'bold', 'size': 11})
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    seed_everything(42) 

    print(">>> 1. 加载数据 (严格匹配 Step 1 输出)...")
    df_tcga_ml = pd.read_csv(f"{DATA_DIR}/dataset_tcga_final.csv", index_col=0)
    df_gse1_ml = pd.read_csv(f"{DATA_DIR}/dataset_gse225846_final.csv", index_col=0)
    df_gse2_ml = pd.read_csv(f"{DATA_DIR}/dataset_gse229571_final.csv", index_col=0)
    
    df_tcga_dl = pd.read_csv(f"{DATA_DIR}/dataset_tcga_top300.csv", index_col=0)
    df_gse1_dl = pd.read_csv(f"{DATA_DIR}/dataset_gse225846_top300.csv", index_col=0)
    df_gse2_dl = pd.read_csv(f"{DATA_DIR}/dataset_gse229571_top300.csv", index_col=0)

    print("\n>>> 2. 预处理...")
    # ML 保持 robust，DL 升级为 0324 验证成功的 robust_log1p
    (X_train_ml, y_train), test_data_ml = preprocess_split(df_tcga_ml, [df_gse1_ml, df_gse2_ml], method='robust')
    X_test1_ml, y_test1 = test_data_ml[0]
    X_test2_ml, y_test2 = test_data_ml[1]
    
    (X_train_dl, y_train_dl), test_data_dl = preprocess_split(df_tcga_dl, [df_gse1_dl, df_gse2_dl], method='robust_log1p')
    X_test1_dl, y_test1_dl = test_data_dl[0] 
    X_test2_dl, y_test2_dl = test_data_dl[1]

    models_dict = {
        "ML": {
            "LR": LogisticRegression(max_iter=3000, C=0.5, solver='liblinear', class_weight='balanced'),
            "LDA": LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            "NaiveBayes": GaussianNB(),
            "KNN": KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan'),
            "RF": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
            "XGB": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=-1, tree_method='hist')
        },
        "DL": {
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, alpha=0.01, random_state=42)
        }
    }
    if HAS_TORCH:
        models_dict["DL"]["CNN"] = SklearnCNN(input_dim=X_train_dl.shape[1], epochs=40, lr=0.001)
    
    plot_data_train, plot_data_gse1, plot_data_gse2 = [], [], []
    sci_results = []
    
    print(f"\n>>> 3. 训练模型并执行 Bootstrap (N={N_BOOTSTRAPS})...")
    for m_type, m_group in models_dict.items():
        X_tr, y_tr = (X_train_ml, y_train) if m_type == "ML" else (X_train_dl, y_train_dl)
        X_t1, y_t1 = (X_test1_ml, y_test1) if m_type == "ML" else (X_test1_dl, y_test1_dl)
        X_t2, y_t2 = (X_test2_ml, y_test2) if m_type == "ML" else (X_test2_dl, y_test2_dl)
        
        for name, base_model in m_group.items():
            print(f"   Training & Bootstrapping {name} ({m_type})...", end="", flush=True)
            model = BalancedBaggingClassifier(base_model, n_estimators=20 if m_type=="ML" else 10)
            model.fit(X_tr, y_tr) 
            
            # ================= 新增：保存训练好的模型为 .pkl 文件 =================
            try:
                model_save_path = f"{OUTPUT_DIR}/{name}_{m_type}_model.pkl"
                joblib.dump(model, model_save_path)
            except Exception as e:
                print(f"\n   [警告] {name} 模型保存失败: {e}")
            # ======================================================================

            p_tr = model.predict_proba(X_tr)[:, 1]
            p_t1 = model.predict_proba(X_t1)[:, 1]
            p_t2 = model.predict_proba(X_t2)[:, 1]
            
            plot_data_train.append((name, y_tr, p_tr))
            plot_data_gse1.append((name, y_t1, p_t1))
            plot_data_gse2.append((name, y_t2, p_t2))
            
            res_tr, auc_tr = evaluate_with_bootstrap(y_tr, p_tr)
            res_t1, auc_t1 = evaluate_with_bootstrap(y_t1, p_t1)
            res_t2, auc_t2 = evaluate_with_bootstrap(y_t2, p_t2)
            
            print(f"\r   {name:12s} | Train AUC: {auc_tr:.3f} | GSE1 AUC: {auc_t1:.3f} | GSE2 AUC: {auc_t2:.3f}")
            
            for ds_name, res in zip(["TCGA_Train", "GSE225846", "GSE229571"], [res_tr, res_t1, res_t2]):
                sci_results.append({
                    "Dataset": ds_name, "Model": name, "Type": m_type, 
                    "AUC (95% CI)": res[0], "Accuracy (95% CI)": res[1], 
                    "Sensitivity (95% CI)": res[2], "Specificity (95% CI)": res[3], 
                    "Precision (95% CI)": res[4], "F1_Score (95% CI)": res[5]
                })

    # ================= 生成 SCI 三线表 =================
    print("\n>>> 4. 生成 SCI 规范化指标表格...")
    df_sci = pd.DataFrame(sci_results)
    
    dataset_order = ["TCGA_Train", "GSE225846", "GSE229571"]
    model_order = list(models_dict["ML"].keys()) + list(models_dict["DL"].keys())
    
    df_sci['Dataset'] = pd.Categorical(df_sci['Dataset'], categories=dataset_order, ordered=True)
    df_sci['Model'] = pd.Categorical(df_sci['Model'], categories=model_order, ordered=True)
    df_sci = df_sci.sort_values(['Dataset', 'Model']).reset_index(drop=True)
    
    df_sci['Dataset'] = df_sci['Dataset'].astype(str)
    df_sci.loc[df_sci.duplicated('Dataset'), 'Dataset'] = ''
    
    sci_csv_path = f"{OUTPUT_DIR}/SCI_metrics_table_bootstrap.csv"
    df_sci.to_csv(sci_csv_path, index=False)
    print(f"     详细指标表格已保存至: {sci_csv_path}")

    # ================= 绘制平滑 ROC 曲线 =================
    print("\n>>> 5. 绘制平滑 ROC 曲线 (Bootstrap Interpolation)...")
    plot_roc_bootstrap_R("TCGA Train", plot_data_train, f"{OUTPUT_DIR}/ROC_TCGA_Train_Smoothed.svg")
    plot_roc_bootstrap_R("GSE225846 Validation", plot_data_gse1, f"{OUTPUT_DIR}/ROC_GSE225846_Smoothed.svg")
    plot_roc_bootstrap_R("GSE229571 Validation", plot_data_gse2, f"{OUTPUT_DIR}/ROC_GSE229571_Smoothed.svg")
    
    # ================= 新增：单独保存 DL 预处理 Scaler 供网页端使用 =================
    print("\n>>> 6. 保存数据预处理 Scaler (供 Shiny 网页端使用)...")
    try:
        # 重新构建 DL 的 scaler 并拟合，以保存为 pkl
        dl_scaler = make_pipeline(RobustScaler(quantile_range=(5, 95)), Normalizer())
        X_train_dl_raw = df_tcga_dl.drop(columns=['target']).values
        # 模拟 robust_log1p 的第一步
        X_train_dl_raw = np.log1p(np.maximum(X_train_dl_raw, 0))
        dl_scaler.fit(X_train_dl_raw)
        
        scaler_save_path = f"{OUTPUT_DIR}/DL_robust_log1p_scaler.pkl"
        joblib.dump(dl_scaler, scaler_save_path)
        print(f"     预处理 Scaler 已保存至: {scaler_save_path}")
    except Exception as e:
        print(f"     [警告] Scaler 保存失败: {e}")
    # ==============================================================================

    print(f"\n✅ 全部完成！平滑后的图片、带 95% CI 的表格以及网页所需的 .pkl 模型均已保存到 {OUTPUT_DIR}")
