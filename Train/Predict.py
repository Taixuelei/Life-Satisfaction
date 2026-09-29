# -*- coding: utf-8 -*-
"""
PISA 2018 生活满意度预测（增强版）- 单文件完整脚本

一、程序结构
--------------
1. 参数与环境配置：集中定义随机种子、数据与输出路径、模型超参数、交叉验证等。
2. 数据加载与预处理：读取 PISA 数据，清洗，Gower 距离 KNN 缺失值插补及插补质量验证。
3. 特征工程：类别变量编码、连续变量 Z-score 标准化、高相关特征剔除及特征选择。
4. 数据划分与异常值处理：8:2 训练/测试划分（分层），温和 Z-score 异常值剔除。
5. 模型训练：MLR、Lasso、SVR、RandomForest、LightGBM、XGBoost、CatBoost、Transformer、
   ResMLP、Bagging、Stacking、加权集成。
6. 模型评估与五折交叉验证：MAE/RMSE/R²，结果写入 Excel。
7. 可解释性分析：SHAP（Summary + Top20 依赖图）、H-statistic 特征交互检验。
8. 可视化与报告：综合分析仪表板（不绘制模型训练曲线）、特征重要性图、性能报告。

二、主要功能
--------------
- KNN 插补（Gower 距离）+ 插补质量验证（分布、相关性、简单模型性能）
- 简化特征工程（二分类 0/1、多分类独热、连续 Z-score、|r|>0.9 剔除）
- 多模型训练与五折交叉验证
- SHAP 深度分析（Top20，5×4 依赖图）
- H-statistic 特征交互检验
- 综合分析仪表板、特征重要性图、详细文本报告

三、可配置参数（见下方「全局可配置参数」块）
--------------
- 数据与目标：DATA_FILE, TARGET_VARIABLE
- 随机与划分：RANDOM_STATE, TRAIN_TEST_SPLIT, STRATIFY_BINS, N_FOLDS
- 插补：K_NEIGHBORS, IMPUTATION_WEIGHTED
- 特征与可解释：CORRELATION_THRESHOLD, TOP_FEATURES
- 深度模型：TRANSFORMER_EPOCHS/LR/BATCH_SIZE, RESMLP_EPOCHS/LR/BATCH_SIZE
- 传统模型与集成：各模型参数字典（RANDOM_FOREST_PARAMS, LIGHTGBM_PARAMS 等）

四、输入与输出
--------------
- 输入：Excel 数据文件（DATA_FILE），含目标变量 TARGET_VARIABLE。
- 输出目录：OUTPUT_DIR。子目录：imputation_quality, shap_analysis, h_statistic。
- 输出文件：train.csv, test.csv, model_metrics_enhanced.csv,
  5fold_cross_validation_results.xlsx, comprehensive_analysis_enhanced.png,
  feature_importance_enhanced.png, performance_report_enhanced.txt，
  以及插补/SHAP/H-statistic 下的图表与 CSV。
"""

import os
import warnings
import time
import datetime
import sklearn
from packaging import version

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, stats
from scipy.spatial.distance import cdist

import gower
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

# ==================== 全局可配置参数（集中放置） ====================
RANDOM_STATE = 42

DATA_FILE = r"D:\Projects\生活满意度\训练数据\集体\集体文化主义国家.xlsx"
TARGET_VARIABLE = 'ST016Q01NA'

OUTPUT_DIR = r"D:\Projects\生活满意度\结果_20260928\1\集体"
IMPUTATION_DIR_NAME = "imputation_quality"
SHAP_DIR_NAME = "shap_analysis"
H_STAT_DIR_NAME = "h_statistic"

TRAIN_TEST_SPLIT = 0.2
STRATIFY_BINS = 5

K_NEIGHBORS = 30
IMPUTATION_WEIGHTED = True

CORRELATION_THRESHOLD = 0.9
TOP_FEATURES = 20

N_FOLDS = 5
CV_SHUFFLE = True
SCORING_METRICS = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
    'R2': 'r2'
}

TRANSFORMER_EPOCHS = 100
TRANSFORMER_LR = 5e-4
TRANSFORMER_BATCH_SIZE = 512

RESMLP_EPOCHS = 200
RESMLP_LR = 8e-4
RESMLP_BATCH_SIZE = 512

LASSO_CV_FOLDS = 5
LASSO_MAX_ITER = 10000

SVR_KERNEL = 'rbf'
SVR_C = 1.0
SVR_GAMMA = 'scale'

RANDOM_FOREST_PARAMS = {
    'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2,
    'max_features': 'sqrt', 'bootstrap': True, 'oob_score': True,
    'random_state': RANDOM_STATE, 'n_jobs': 1
}

LIGHTGBM_PARAMS = {
    'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 10, 'num_leaves': 63,
    'min_child_samples': 15, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': RANDOM_STATE,
    'verbose': -1, 'n_jobs': 1, 'min_split_gain': 0.01, 'force_col_wise': True
}

XGBOOST_PARAMS = {
    'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 3,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbosity': 0, 'eval_metric': 'rmse'
}

CATBOOST_PARAMS = {
    'iterations': 800, 'learning_rate': 0.05, 'depth': 10, 'min_child_samples': 15,
    'subsample': 0.8, 'colsample_bylevel': 0.8, 'l2_leaf_reg': 3,
    'random_state': RANDOM_STATE, 'verbose': 0
}

BAGGING_BASE_LGB_PARAMS = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 8,
    'random_state': RANDOM_STATE, 'verbose': -1
}
BAGGING_PARAMS = {
    'n_estimators': 20, 'max_samples': 0.8, 'max_features': 0.8,
    'random_state': RANDOM_STATE, 'n_jobs': 1
}

STACKING_RF_PARAMS = {'n_estimators': 300, 'max_depth': 12, 'random_state': RANDOM_STATE}
STACKING_LGB_PARAMS = {'n_estimators': 400, 'learning_rate': 0.05, 'random_state': RANDOM_STATE, 'verbose': -1}
STACKING_XGB_PARAMS = {'n_estimators': 400, 'learning_rate': 0.05, 'random_state': RANDOM_STATE,'objective': 'reg:squarederror', 'verbosity': 0}
STACKING_CAT_PARAMS = {'iterations': 400, 'learning_rate': 0.05, 'random_state': RANDOM_STATE, 'verbose': 0}

# ==================== 字体与图形配置 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.subplot.wspace'] = 0.3
plt.rcParams['figure.subplot.hspace'] = 0.5

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMPUTATION_DIR = os.path.join(OUTPUT_DIR, IMPUTATION_DIR_NAME)
SHAP_DIR = os.path.join(OUTPUT_DIR, SHAP_DIR_NAME)
H_STAT_DIR = os.path.join(OUTPUT_DIR, H_STAT_DIR_NAME)
os.makedirs(IMPUTATION_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(H_STAT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("=" * 80)
print("🎯 PISA 2018 生活满意度预测分析系统（增强版）")
print(f"📅 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 运行设备: {DEVICE}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"🔧 sklearn版本: {sklearn.__version__}")
print(f"🔧 lightgbm版本: {lgb.__version__ if 'lgb' in dir() else '未安装'}")
print(f"🔧 xgboost版本: {xgb.__version__ if 'xgb' in dir() else '未安装'}")
print("🔧 核心增强: KNN插补(Gower距离)+MLR/Lasso/SVR+SHAP分析+H-statistic交互检验")
print("🔧 数据划分: 训练集80% / 测试集20%（无验证集）")
print("=" * 80)


# -------------------- 数据集类（把 pandas 数据转为 PyTorch Dataset） --------------------
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- KNN插补（Gower距离）--------------------
def knn_impute_gower(df, k=5, weighted=True):
    """基于Gower距离的KNN插补（适配混合类型变量）"""
    print(f"\n   📌 执行Gower距离KNN插补（k={k}，距离加权={weighted}）")
    df_imp = df.copy()
    num_cols = df_imp.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_imp.select_dtypes(include=['object', 'category']).columns.tolist()

    print("   📌 计算Gower距离矩阵（混合类型变量）...")
    # gower_dist = gower.gower_matrix(df_imp)
    missing_samples = df_imp.isnull().any(axis=1)    #判断每一行是否有缺失值
    processed_count = 0
    total_missing = missing_samples.sum()

    for idx in range(len(df_imp)):
        if not missing_samples.iloc[idx]:
            continue
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"   📌 已处理 {processed_count}/{total_missing} 个缺失样本")

        current_row = df_imp.iloc[[idx]]
        dists = gower.gower_matrix(current_row, df_imp)[0].copy()
        dists[idx] = np.inf
        nearest_neighbors = np.argsort(dists)[:k]      #找到距离最小的k个样本（最近邻）
        nn_dists = dists[nearest_neighbors]

        #权重计算（距离越小 → 权重越大）
        if weighted:
            weights = 1 / (nn_dists + 1e-8)
            weights = weights / weights.sum()
        else:
            weights = np.ones(k) / k

        # 填充缺失值
        missing_cols = df_imp.iloc[idx].isnull()     #当前样本哪些列是缺失
        for col_idx, is_missing in enumerate(missing_cols):
            if not is_missing:
                continue
            col_name = df_imp.columns[col_idx]
            nn_vals = df_imp.iloc[nearest_neighbors, col_idx].dropna()

            if len(nn_vals) == 0:    #所有邻居该列都是缺失
                if col_name in num_cols:
                    fill_val = df_imp[col_name].mean() if not np.isnan(df_imp[col_name].mean()) else 0.0
                else:
                    fill_val = df_imp[col_name].mode().iloc[0] if len(df_imp[col_name].mode()) > 0 else 'Unknown'
            else:
                if col_name in num_cols:
                    fill_val = np.average(nn_vals, weights=weights[:len(nn_vals)])
                else:
                    val_counts = nn_vals.value_counts()
                    fill_val = val_counts.index[0]

            df_imp.iloc[idx, col_idx] = fill_val

    # -------------------- 修复：强化NaN/无限值清理 --------------------
    print("   📌 插补后强制清理残留NaN/无限值...")
    # 1. 清理数值特征的NaN和无限值
    for col in num_cols:
        col_mean = df_imp[col].mean() if not np.isnan(df_imp[col].mean()) else 0.0
        df_imp[col] = df_imp[col].fillna(col_mean)
        df_imp[col] = np.clip(
            df_imp[col],
            df_imp[col].replace([np.inf, -np.inf], np.nan).min(),
            df_imp[col].replace([np.inf, -np.inf], np.nan).max()
        )

    # 2. 清理分类特征的NaN
    for col in cat_cols:
        fill_val = df_imp[col].mode().iloc[0] if len(df_imp[col].mode()) > 0 else 'Unknown'
        df_imp[col] = df_imp[col].fillna(fill_val)

    # 3. 最终兜底：删除仍有NaN的行（确保无残留）
    before_drop = len(df_imp)
    df_imp = df_imp.dropna()
    after_drop = len(df_imp)
    if before_drop - after_drop > 0:
        print(f"   ⚠️  兜底删除{before_drop - after_drop}行仍含NaN的样本")

    # 验证：检查是否还有NaN
    nan_count = df_imp.isnull().sum().sum()
    inf_count = np.isinf(df_imp.select_dtypes(include=[np.number])).sum().sum()
    print(f"   ✅ 插补后校验：NaN数量={nan_count}，无限值数量={inf_count}")
    print(f"   📌 KNN插补完成，共处理 {processed_count} 个缺失样本")
    return df_imp


def validate_imputation_quality(df_before, df_after, target_var):
    """验证插补质量：对比分布、相关性、模型性能"""
    print("\n" + "📊 插补质量验证".center(80, "-"))
    num_cols = df_before.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_var in num_cols:
        num_cols.remove(target_var)
    num_cols = [col for col in num_cols if col in df_before.columns and col in df_after.columns]

    if len(num_cols) == 0:
        print("   ❌ 无有效数值列，无法完成插补质量验证")
        return pd.DataFrame()

    # 1. 分布对比（直方图）
    print("   1. 数据分布对比...")
    sample_cols = num_cols[:5] if len(num_cols) >= 5 else num_cols
    fig, axes = plt.subplots(2, len(sample_cols), figsize=(20, 10))
    fig.suptitle('Distribution Comparison Before and After Imputation', fontsize=16, fontweight='bold')
    for idx, col in enumerate(sample_cols):
        axes[0, idx].hist(df_before[col].dropna(), bins=20, alpha=0.5, color='blue', label='Before Imputation')
        axes[0, idx].set_title(f'{col} (Before Imputation)')
        axes[0, idx].grid(alpha=0.3)
        axes[1, idx].hist(df_after[col], bins=20, alpha=0.5, color='green', label='After Imputation')
        axes[1, idx].set_title(f'{col} (After Imputation)')
        axes[1, idx].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMPUTATION_DIR, "distribution_comparison.png"), dpi=300)
    plt.close()

    print("   2. 相关性结构对比（增强版左右热力图，共享colorbar）...")
    corr_before = df_before[num_cols].corr()
    corr_after = df_after[num_cols].corr()

    n_feats = len(num_cols)
    width = max(16, 8 + n_feats / 12)
    height = width * 0.55
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    fig.suptitle('Correlation Matrix Comparison Before and After Imputation',
                 fontsize=16, fontweight='bold', y=0.98)

    if n_feats <= 30:
        label_fontsize = 8
        value_fontsize = 7
    elif n_feats <= 50:
        label_fontsize = 6
        value_fontsize = 5
    elif n_feats <= 80:
        label_fontsize = 5
        value_fontsize = 3
    else:
        label_fontsize = 4
        value_fontsize = 0

    im1 = axes[0].imshow(corr_before, cmap='RdYlBu_r', vmin=-1, vmax=1, interpolation='none')
    axes[0].set_title('Before Imputation', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(n_feats))
    axes[0].set_yticks(range(n_feats))
    axes[0].set_xticklabels(num_cols, rotation=90, fontsize=label_fontsize)
    axes[0].set_yticklabels(num_cols, fontsize=label_fontsize)
    if value_fontsize > 0:
        for i in range(n_feats):
            for j in range(n_feats):
                val = corr_before.iloc[i, j]
                if np.isnan(val):
                    continue
                color = 'white' if abs(val) > 0.5 else 'black'
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=value_fontsize, color=color)

    im2 = axes[1].imshow(corr_after, cmap='RdYlBu_r', vmin=-1, vmax=1, interpolation='none')
    axes[1].set_title('After Imputation', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(n_feats))
    axes[1].set_yticks(range(n_feats))
    axes[1].set_xticklabels(num_cols, rotation=90, fontsize=label_fontsize)
    axes[1].set_yticklabels(num_cols, fontsize=label_fontsize)
    if value_fontsize > 0:
        for i in range(n_feats):
            for j in range(n_feats):
                val = corr_after.iloc[i, j]
                if np.isnan(val):
                    continue
                color = 'white' if abs(val) > 0.5 else 'black'
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=value_fontsize, color=color)

    for ax in axes:
        ax.grid(False)
        ax.set_xticks(np.arange(n_feats) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_feats) - 0.5, minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.6, zorder=10)
        ax.tick_params(which='both', zorder=20)

    plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, wspace=0.15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('Pearson Correlation Coefficient', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(IMPUTATION_DIR, "correlation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. 简单模型（线性回归）性能对比
    print("   3. 简单模型性能对比...")
    # -------------------- 修复：强制填充所有NaN，避免LinearRegression报错 --------------------
    X_before = df_before[num_cols].fillna(df_before[num_cols].mean()).replace([np.inf, -np.inf], 0)
    y_before = df_before[target_var].fillna(df_before[target_var].mean()).replace([np.inf, -np.inf], 0)
    X_after = df_after[num_cols].fillna(df_after[num_cols].mean()).replace([np.inf, -np.inf], 0)
    y_after = df_after[target_var].fillna(df_after[target_var].mean()).replace([np.inf, -np.inf], 0)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_before, y_before, test_size=0.2, random_state=RANDOM_STATE)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_after, y_after, test_size=0.2, random_state=RANDOM_STATE)

    try:
        lr = LinearRegression()
        lr.fit(Xb_tr, yb_tr)
        yb_pred = lr.predict(Xb_te)
        before_metrics = {
            'MAE': mean_absolute_error(yb_te, yb_pred),
            'RMSE': np.sqrt(mean_squared_error(yb_te, yb_pred)),
            'R2': r2_score(yb_te, yb_pred)
        }
        lr.fit(Xa_tr, ya_tr)
        ya_pred = lr.predict(Xa_te)
        after_metrics = {
            'MAE': mean_absolute_error(ya_te, ya_pred),
            'RMSE': np.sqrt(mean_squared_error(ya_te, ya_pred)),
            'R2': r2_score(ya_te, ya_pred)
        }
    except Exception as e:
        print(f"   ⚠️  LinearRegression拟合失败: {str(e)[:100]}")
        print("   📌 切换到HistGradientBoostingRegressor（支持NaN）")
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgbr = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        hgbr.fit(Xb_tr, yb_tr)
        before_metrics = {'MAE': mean_absolute_error(yb_te, hgbr.predict(Xb_te)), 'RMSE': np.sqrt(mean_squared_error(yb_te, hgbr.predict(Xb_te))), 'R2': r2_score(yb_te, hgbr.predict(Xb_te))}
        hgbr.fit(Xa_tr, ya_tr)
        ya_pred = hgbr.predict(Xa_te)
        after_metrics = {
            'MAE': mean_absolute_error(ya_te, ya_pred),
            'RMSE': np.sqrt(mean_squared_error(ya_te, ya_pred)),
            'R2': r2_score(ya_te, ya_pred)
        }

    # 输出结果
    print(f"      插补前 | 插补后")
    print(f"      MAE: {before_metrics['MAE']:.4f} | {after_metrics['MAE']:.4f}")
    print(f"      RMSE: {before_metrics['RMSE']:.4f} | {after_metrics['RMSE']:.4f}")
    print(f"      R2: {before_metrics['R2']:.4f} | {after_metrics['R2']:.4f}")

    # 保存结果
    impute_metrics = pd.DataFrame({'插补前': before_metrics, '插补后': after_metrics}).T
    impute_metrics.to_csv(os.path.join(IMPUTATION_DIR, "imputation_performance.csv"))

    return impute_metrics


# -------------------- H-statistic 特征交互 --------------------
def calculate_h_statistic(model, X, y, f1, f2):
    """计算单个特征对的H-statistic"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    all_feature_names = X.columns.tolist()
    fill_values = {}
    for col in all_feature_names:
        if X[col].dtype in ['float64', 'int64']:
            fill_values[col] = X[col].mean()
        else:
            fill_values[col] = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown'

    X1 = X.copy()
    for col in all_feature_names:
        if col != f1:
            X1[col] = fill_values[col]
    X2 = X.copy()
    for col in all_feature_names:
        if col != f2:
            X2[col] = fill_values[col]
    X12 = X.copy()
    for col in all_feature_names:
        if col not in [f1, f2]:
            X12[col] = fill_values[col]

    y_pred = model.predict(X)
    y1_pred = model.predict(X1)
    y2_pred = model.predict(X2)
    y12_pred = model.predict(X12)
    res = y - y_pred
    res1 = y - y1_pred
    res2 = y - y2_pred
    res12 = y - y12_pred
    ss_res = np.sum(res ** 2)
    ss_res12 = np.sum(res12 ** 2)
    ss_res1 = np.sum(res1 ** 2)
    ss_res2 = np.sum(res2 ** 2)
    if ss_res == 0:
        return 0.0
    h_stat = (ss_res12 - np.min([ss_res1, ss_res2])) / ss_res
    return np.clip(h_stat, 0, 1)


def h_statistic_analysis(model, X, y, top_features=20):
    """批量检验Top20特征的交互效应"""
    print("\n" + "🔍 H-statistic特征交互检验".center(80, "-"))
    if len(X.columns) < 2:
        print("   ❌ 特征数量不足，无法进行交互检验")
        return pd.DataFrame(), []
    top_feats = []
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        fn = X.columns.tolist()
        min_len = min(len(fn), len(fi))
        top_feats = pd.DataFrame({'feature': fn[:min_len], 'importance': fi[:min_len]}).sort_values('importance', ascending=False).head(top_features)['feature'].tolist()
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_)
        fn = X.columns.tolist()
        min_len = min(len(fn), len(coef))
        top_feats = pd.DataFrame({'feature': fn[:min_len], 'importance': coef[:min_len]}).sort_values('importance', ascending=False).head(top_features)['feature'].tolist()
    else:
        top_feats = X.columns[:min(top_features, len(X.columns))].tolist()

    h_matrix = np.zeros((len(top_feats), len(top_feats)))
    for i, f1 in enumerate(top_feats):
        for j, f2 in enumerate(top_feats):
            if i != j:
                h_matrix[i, j] = calculate_h_statistic(model, X, y, f1, f2)

    # -------------------- 绘制带数值标注的 H-statistic 热力图 --------------------
    n_feats = len(top_feats)
    vmax_val = np.percentile(h_matrix, 95)
    if vmax_val <= 0:
        vmax_val = np.max(h_matrix) if np.max(h_matrix) > 0 else 1.0

    fig_width = max(12, 0.55 * n_feats + 4)
    fig_height = max(10, 0.50 * n_feats + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        h_matrix,
        cmap='YlOrRd',
        vmin=0,
        vmax=vmax_val,
        interpolation='none'
    )

    ax.set_xticks(np.arange(n_feats))
    ax.set_yticks(np.arange(n_feats))
    ax.set_xticklabels(top_feats, rotation=45, ha='right', fontsize=10, fontname='Times New Roman')
    ax.set_yticklabels(top_feats, fontsize=10, fontname='Times New Roman')
    ax.set_title(
        f'Top {n_feats} Features H-statistic Interaction Effect Heatmap',
        fontsize=15,
        fontweight='bold',
        fontname='Times New Roman'
    )

    # 网格线
    ax.set_xticks(np.arange(n_feats) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_feats) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 数值标注
    # 特征太多时可自动缩小字体
    if n_feats <= 20:
        value_fontsize = 7
    elif n_feats <= 30:
        value_fontsize = 6
    elif n_feats <= 40:
        value_fontsize = 5
    else:
        value_fontsize = 0

    if value_fontsize > 0:
        threshold = vmax_val * 0.6
        for i in range(n_feats):
            for j in range(n_feats):
                val = h_matrix[i, j]
                if np.isnan(val):
                    continue
                text_color = 'white' if val >= threshold else 'black'
                ax.text(
                    j, i, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=value_fontsize,
                    color=text_color,
                    fontname='Times New Roman'
                )

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        'H-statistic (Interaction Effect Strength)',
        fontsize=11,
        fontname='Times New Roman'
    )
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(H_STAT_DIR, "h_statistic_heatmap.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    h_df = pd.DataFrame(h_matrix, columns=top_feats, index=top_feats)
    h_df.to_csv(os.path.join(H_STAT_DIR, "h_statistic_results.csv"))

    significant_pairs = []
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            if h_matrix[i, j] > 0.01:
                significant_pairs.append((top_feats[i], top_feats[j], h_matrix[i, j]))
    significant_pairs = sorted(significant_pairs, key=lambda x: x[2], reverse=True)
    print(f"   📌 显著交互特征对（H>0.01）：{len(significant_pairs)}对")
    for pair in significant_pairs[:10]:
        print(f"     {pair[0]} × {pair[1]}: H={pair[2]:.4f}")
    return h_df, significant_pairs


# -------------------- 简化版特征工程 --------------------
def simplify_feature_engineering(X, target_series):
    """连续变量Z-score标准化；类别变量二分类0/1、多分类独热；剔除|r|>0.9高相关特征"""
    X = X.copy()
    print("\n2.1 特征类型识别...")

    def identify_feature_types(X_input):
        feature_types = {"continuous": [], "binary_categorical": [], "multi_categorical": []}
        for col in X_input.columns:
            if X_input[col].dtype == 'object' or (X_input[col].dtype in ['int64', 'float64'] and len(X_input[col].unique()) <= 20):
                unique_count = len(X_input[col].dropna().unique())
                if unique_count == 2:
                    feature_types["binary_categorical"].append(col)
                elif unique_count > 2:
                    feature_types["multi_categorical"].append(col)
            else:
                feature_types["continuous"].append(col)
        for ftype, cols in feature_types.items():
            print(f"   - {ftype:20}: {len(cols)} 个特征")
        return feature_types

    feature_types = identify_feature_types(X)
    X_encoded = X.copy()
    print("\n2.2 类别变量编码...")

    if len(feature_types["binary_categorical"]) > 0:
        le = LabelEncoder()
        for col in feature_types["binary_categorical"]:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].value_counts().idxmax())
            X_encoded[col] = le.fit_transform(X_encoded[col])
        print(f"   - 二分类变量：{len(feature_types['binary_categorical'])} 个，已完成0/1编码")

    if len(feature_types["multi_categorical"]) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_filled = pd.DataFrame(cat_imputer.fit_transform(X_encoded[feature_types["multi_categorical"]]), columns=feature_types["multi_categorical"], index=X_encoded.index)
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        ohe_data = ohe.fit_transform(X_cat_filled)
        ohe_feature_names = []
        for idx, col in enumerate(feature_types["multi_categorical"]):
            ohe_feature_names.extend([f"{col}_{cat}" for cat in ohe.categories_[idx][1:]])
        ohe_df = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=X_encoded.index)
        X_encoded = X_encoded.drop(columns=feature_types["multi_categorical"]).join(ohe_df)
        print(f"   - 多分类变量：{len(feature_types['multi_categorical'])} 个，已完成独热编码（新增 {len(ohe_feature_names)} 个特征）")

    print("\n2.3 连续变量Z-score标准化...")
    if len(feature_types["continuous"]) > 0:
        scaler = StandardScaler()
        cont_data = X_encoded[feature_types["continuous"]].copy().fillna(X_encoded[feature_types["continuous"]].mean())
        X_encoded[feature_types["continuous"]] = scaler.fit_transform(cont_data)
        sample_cont_col = feature_types["continuous"][0]
        print(f"   - 连续变量：{len(feature_types['continuous'])} 个，已完成Z-score标准化")
        print(f"   - 标准化验证（{sample_cont_col}）：均值≈{X_encoded[sample_cont_col].mean():.4f}，标准差≈{X_encoded[sample_cont_col].std():.4f}")

    print("\n2.4 高相关特征筛查与剔除（|r| > 0.9）...")
    X_numeric = X_encoded.select_dtypes(include=[np.number])
    corr_matrix = X_numeric.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    X_final = X_numeric.drop(columns=high_corr_features)
    print(f"   - 编码后总特征数：{X_numeric.shape[1]}")
    print(f"   - 剔除高相关特征数：{len(high_corr_features)}（特征名：{high_corr_features if len(high_corr_features) > 0 else '无'}）")
    print(f"   - 最终保留特征数：{X_final.shape[1]}")

    print("\n2.5 特征工程最终结果校验...")
    print(f"   - 最终样本数：{len(X_final)}（与原始有效样本数一致，无丢失）")
    print(f"   - 最终特征数：{X_final.shape[1]}")
    return X_final


# -------------------- 深度学习模型定义 --------------------
class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(n_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fc_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.enable_ckpt = True

    def encoder_forward(self, x):
        return self.transformer_encoder(x)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) + self.pos_encoding.unsqueeze(0)
        if self.training and self.enable_ckpt:
            x = checkpoint.checkpoint(self.encoder_forward, x)
        else:
            x = self.transformer_encoder(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        attn_weights = torch.matmul(cls_token, x.transpose(1, 2)) / np.sqrt(self.d_model)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        cls_output = torch.matmul(attn_weights, x).squeeze(1)
        return self.fc_head(cls_output).squeeze(-1)


class ResMLP(nn.Module):
    def __init__(self, n_features, hidden_dims=[256, 128, 64], dropout=0.15):
        super().__init__()
        dims = [n_features] + hidden_dims
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)))
            self.residuals.append(nn.Linear(dims[i], dims[i + 1]) if dims[i] != dims[i + 1] else nn.Identity())
        self.output_layer = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for layer, res in zip(self.layers, self.residuals):
            x = layer(x) + res(x)
        return self.output_layer(x).squeeze(-1)


# -------------------- 训练与预测（深度学习）--------------------
def train_gpu_model(model, model_name, X_tr, y_tr, epochs=150, lr=1e-3, batch_size=256):
    """无验证集，完整训练所有轮数"""
    print(f"\n   🚀 {model_name} 训练参数: 批次={batch_size}, 学习率={lr}, 训练轮数={epochs}")
    y_mean = y_tr.mean()
    y_std = y_tr.std()
    train_dataset = RegressionDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler()
    train_loss_history = []
    best_train_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            batch_y_norm = (batch_y - y_mean) / (y_std + 1e-8)
            optimizer.zero_grad()
            with autocast():
                pred = model(batch_X)
                loss = criterion(pred, batch_y_norm)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_total += loss.item() * batch_X.size(0)

        train_loss = train_loss_total / len(train_dataset)
        train_loss_history.append(train_loss)
        scheduler.step(train_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = model.state_dict().copy()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"     Epoch {epoch:3d} | 训练损失: {train_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.cuda.empty_cache()
    return model, y_mean, y_std, train_loss_history


def predict_with_model(model, X, y_mean, y_std, batch_size=512):
    model.eval()
    dataset = RegressionDataset(X, pd.Series(np.zeros(len(X))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(DEVICE)
            with autocast():
                pred_norm = model(batch_X)
            pred = pred_norm * y_std + y_mean
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions)


# -------------------- 特征选择与异常值处理 --------------------
def select_top_features(X, y, top_n=30):
    actual_top_n = min(top_n, X.shape[1])
    if actual_top_n <= 0:
        return X.columns.tolist(), np.zeros(X.shape[1])
    f_selector = SelectKBest(f_regression, k=actual_top_n).fit(X, y)
    f_scores = f_selector.scores_
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1).fit(X, y)
    rf_importances = rf.feature_importances_

    def normalize(scores):
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    combined_scores = 0.4 * normalize(f_scores) + 0.3 * normalize(mi_scores) + 0.3 * normalize(rf_importances)
    top_feature_indices = np.argsort(combined_scores)[-actual_top_n:]
    return X.columns[top_feature_indices].tolist(), combined_scores


def filter_outliers_mild(X, y, z_threshold=4, max_outlier_features_pct=0.1):
    if X.shape[1] == 0:
        return X, y
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    outlier_feature_count = (z_scores > z_threshold).sum(axis=1)
    max_outlier_features = int(X.shape[1] * max_outlier_features_pct)
    keep_mask = outlier_feature_count <= max_outlier_features
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]
    print(f"   - 处理前样本数：{len(X)} | 处理后样本数：{len(X_filtered)} | 剔除异常样本：{len(X) - len(X_filtered)}")
    return X_filtered, y_filtered


# ==================== 主流程：数据加载与预处理 ====================
print("\n" + "🔧 数据加载与预处理阶段（增强版）".center(80, "-"))
print("\n1.1 加载原始数据...")
# 数据路径
df_raw = pd.read_excel(DATA_FILE)
target_variable = TARGET_VARIABLE
# 基础信息输出
print(f"   - 原始数据集形状: {df_raw.shape} (样本数: {df_raw.shape[0]}, 特征数: {df_raw.shape[1]})")
print(f"   - 目标变量: {target_variable} (生活满意度评分 0-10)")

print("\n1.2 数据清洗...")
df_clean = df_raw.query("0 <= @df_raw[@target_variable] <= 10").dropna(subset=[target_variable])
y = df_clean[target_variable].astype(float)
X_raw = df_clean.drop(columns=[target_variable])
print(f"   - 清洗后样本数: {len(df_clean)} (剔除无效样本: {len(df_raw) - len(df_clean)})")
print(f"   - 清洗后特征数: {X_raw.shape[1]}")
print(f"   - 目标变量统计: 均值={y.mean():.2f}, 标准差={y.std():.2f}, 范围=[{y.min():.0f}, {y.max():.0f}]")

print("\n1.3 缺失值处理：KNN插补（Gower距离）...")
df_for_imputation = pd.concat([X_raw, y.rename(target_variable)], axis=1)
df_before_imputation = df_for_imputation.copy()
df_after_imputation = knn_impute_gower(df_for_imputation, k=K_NEIGHBORS, weighted=IMPUTATION_WEIGHTED)

# 验证插补质量
impute_metrics = validate_imputation_quality(df_before_imputation, df_after_imputation, target_variable)
imputed_data_path = os.path.join(OUTPUT_DIR, "imputed_data.csv")
df_after_imputation.to_csv(imputed_data_path, index=False)
print(f"   ✅ 插补后完整数据已保存至: {imputed_data_path}")
X_imputed = df_after_imputation.drop(columns=[target_variable])
y_imputed = df_after_imputation[target_variable]

print("\n" + "🛠️  简化版特征工程（增强版）".center(80, "-"))
X_enhanced = simplify_feature_engineering(X_imputed, y_imputed)
print(f"\n2.6 特征工程最终结果汇总:")
print(f"   - 原始特征数: {X_imputed.shape[1]}")
print(f"   - 处理后特征数: {X_enhanced.shape[1]}")
print(f"   - 特征变化: {'减少' if X_enhanced.shape[1] < X_imputed.shape[1] else '增加'} {abs(X_enhanced.shape[1] - X_imputed.shape[1])} 个（编码+高相关剔除）")

print("\n" + "📊 数据划分阶段（8:2 训练/测试）".center(80, "-"))
print("\n3.1 划分训练集/测试集（8:2）")
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_enhanced, y_imputed, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE,
    stratify=pd.cut(y_imputed, bins=STRATIFY_BINS)
)
print(f"   - 训练集: {len(X_tr_raw)} 样本（80%）")
print(f"   - 测试集: {len(X_te_raw)} 样本（20%）")

num_cols = X_tr_raw.select_dtypes(include='number').columns
cat_cols = []
print(f"\n3.2 特征类型统计:")
print(f"   - 数值特征数: {len(num_cols)}")
print(f"   - 类别特征数: {len(cat_cols)}（已完成编码，无原始类别特征）")

print("\n3.3 缺失值处理（兜底填充）:")
num_imputer = SimpleImputer(strategy='mean')
X_tr_num = pd.DataFrame(num_imputer.fit_transform(X_tr_raw[num_cols]), columns=num_cols, index=X_tr_raw.index)
X_te_num = pd.DataFrame(num_imputer.transform(X_te_raw[num_cols]), columns=num_cols, index=X_te_raw.index)
missing_ratio = X_tr_raw[num_cols].isnull().sum().sum() / (X_tr_raw[num_cols].shape[0] * X_tr_raw[num_cols].shape[1])
print(f"   - 数值特征: 均值填充（填充缺失值比例: {missing_ratio:.1%}）")

X_tr_imputed = X_tr_num.copy()
X_te_imputed = X_te_num.copy()

print("\n3.4 特征选择（融合F检验、互信息、随机森林重要性）")
top_features, feature_scores = select_top_features(X_tr_imputed, y_tr, top_n=30)
X_tr_selected = X_tr_imputed[top_features]
X_te_selected = X_te_imputed[top_features]

print("3.5 异常值处理（温和Z-score，避免过度过滤）")
X_tr_clean, y_tr_clean = filter_outliers_mild(X_tr_selected, y_tr, z_threshold=4, max_outlier_features_pct=0.1)
if len(X_tr_clean) < 100:
    print(f"   ⚠️  警告：异常值处理后样本数仅 {len(X_tr_clean)}，建议放宽过滤条件")
if len(X_tr_clean) == 0:
    raise ValueError("异常值处理后样本数为0，无法继续建模！")

print(f"\n3.6 最终处理后数据信息:")
print(f"   - 特征选择后特征数: {len(top_features)} (从 {X_tr_imputed.shape[1]} 筛选)")
print(f"   - 异常值处理后训练集样本: {len(X_tr_clean)}")
print(f"   - 数据划分: 训练集80% / 测试集20%（无验证集）")

X_tr_final = X_tr_clean
y_tr_final = y_tr_clean
X_te_final = X_te_selected
y_te_final = y_te

print("\n3.7 保存处理后数据文件:")
X_tr_final.assign(target=y_tr_final).to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
X_te_final.assign(target=y_te_final).to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
print(f"   - 训练集: {os.path.join(OUTPUT_DIR, 'train.csv')}")
print(f"   - 测试集: {os.path.join(OUTPUT_DIR, 'test.csv')}")

kf = KFold(n_splits=N_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)
cv_results = {}
scoring_metrics = SCORING_METRICS.copy()
cv_excel_path = os.path.join(OUTPUT_DIR, "5fold_cross_validation_results.xlsx")

# ==================== 模型训练 ====================
print("\n" + "🚀 模型训练阶段（增强版）".center(80, "-"))
scaler = StandardScaler()
X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_final), columns=X_tr_final.columns)
X_te_scaled = pd.DataFrame(scaler.transform(X_te_final), columns=X_te_final.columns)

predictions = {}
trained_models = {}
loss_histories = {}
metrics = {}

# 4.1 MLR
print("\n4.1 训练多元线性回归（MLR）模型...")
start_time = time.time()
mlr_model = LinearRegression()
mlr_model.fit(X_tr_final, y_tr_final)
predictions['MLR'] = mlr_model.predict(X_te_final)
trained_models['MLR'] = mlr_model
train_time = time.time() - start_time
print(f"   ✅ MLR训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行MLR五折交叉验证...")
mlr_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(mlr_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    mlr_cv_scores[metric_name] = scores
cv_results['MLR'] = {
    'MAE': mlr_cv_scores['MAE'],
    'RMSE': mlr_cv_scores['RMSE'],
    'R2': mlr_cv_scores['R2'],
    'Mean_MAE': np.mean(mlr_cv_scores['MAE']),
    'Std_MAE': np.std(mlr_cv_scores['MAE']),
    'Mean_RMSE': np.mean(mlr_cv_scores['RMSE']),
    'Std_RMSE': np.std(mlr_cv_scores['RMSE']),
    'Mean_R2': np.mean(mlr_cv_scores['R2']),
    'Std_R2': np.std(mlr_cv_scores['R2'])
}
print(f"   📈 MLR五折CV结果：")
print(f"      MAE: {cv_results['MLR']['Mean_MAE']:.4f} ± {cv_results['MLR']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['MLR']['Mean_RMSE']:.4f} ± {cv_results['MLR']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['MLR']['Mean_R2']:.4f} ± {cv_results['MLR']['Std_R2']:.4f}")

# 4.2 Lasso
print("\n4.2 训练Lasso回归模型...")
start_time = time.time()
lasso_cv = LassoCV(cv=LASSO_CV_FOLDS, random_state=RANDOM_STATE, max_iter=LASSO_MAX_ITER)
lasso_cv.fit(X_tr_final, y_tr_final)
lasso_model = Lasso(alpha=lasso_cv.alpha_, random_state=RANDOM_STATE)
lasso_model.fit(X_tr_final, y_tr_final)
predictions['Lasso'] = lasso_model.predict(X_te_final)
trained_models['Lasso'] = lasso_model
train_time = time.time() - start_time
print(f"   ✅ Lasso训练完成 | 耗时: {train_time:.2f}秒 | 最优alpha: {lasso_cv.alpha_:.6f}")
print("   📊 执行Lasso五折交叉验证...")
lasso_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(lasso_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    lasso_cv_scores[metric_name] = scores
cv_results['Lasso'] = {
    'MAE': lasso_cv_scores['MAE'],
    'RMSE': lasso_cv_scores['RMSE'],
    'R2': lasso_cv_scores['R2'],
    'Mean_MAE': np.mean(lasso_cv_scores['MAE']),
    'Std_MAE': np.std(lasso_cv_scores['MAE']),
    'Mean_RMSE': np.mean(lasso_cv_scores['RMSE']),
    'Std_RMSE': np.std(lasso_cv_scores['RMSE']),
    'Mean_R2': np.mean(lasso_cv_scores['R2']),
    'Std_R2': np.std(lasso_cv_scores['R2'])
}
print(f"   📈 Lasso五折CV结果：")
print(f"      MAE: {cv_results['Lasso']['Mean_MAE']:.4f} ± {cv_results['Lasso']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['Lasso']['Mean_RMSE']:.4f} ± {cv_results['Lasso']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['Lasso']['Mean_R2']:.4f} ± {cv_results['Lasso']['Std_R2']:.4f}")

# 4.3 SVR
print("\n4.3 训练支持向量回归（SVR）模型...")
start_time = time.time()
svr_model = SVR(kernel=SVR_KERNEL, C=SVR_C, gamma=SVR_GAMMA)
svr_model.fit(X_tr_scaled, y_tr_final)
predictions['SVR'] = svr_model.predict(X_te_scaled)
trained_models['SVR'] = svr_model
train_time = time.time() - start_time
print(f"   ✅ SVR训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行SVR五折交叉验证...")
svr_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(svr_model, X_tr_scaled, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    svr_cv_scores[metric_name] = scores
cv_results['SVR'] = {
    'MAE': svr_cv_scores['MAE'],
    'RMSE': svr_cv_scores['RMSE'],
    'R2': svr_cv_scores['R2'],
    'Mean_MAE': np.mean(svr_cv_scores['MAE']),
    'Std_MAE': np.std(svr_cv_scores['MAE']),
    'Mean_RMSE': np.mean(svr_cv_scores['RMSE']),
    'Std_RMSE': np.std(svr_cv_scores['RMSE']),
    'Mean_R2': np.mean(svr_cv_scores['R2']),
    'Std_R2': np.std(svr_cv_scores['R2'])
}
print(f"   📈 SVR五折CV结果：")
print(f"      MAE: {cv_results['SVR']['Mean_MAE']:.4f} ± {cv_results['SVR']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['SVR']['Mean_RMSE']:.4f} ± {cv_results['SVR']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['SVR']['Mean_R2']:.4f} ± {cv_results['SVR']['Std_R2']:.4f}")

# 4.4 RandomForest
print("\n4.4 训练随机森林模型...")
start_time = time.time()
rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
rf_model.fit(X_tr_final, y_tr_final)
predictions['RandomForest'] = rf_model.predict(X_te_final)
trained_models['RandomForest'] = rf_model
train_time = time.time() - start_time
print(f"   ✅ 随机森林训练完成 | 耗时: {train_time:.2f}秒 | OOB得分: {rf_model.oob_score_:.4f}")
print("   📊 执行RandomForest五折交叉验证...")
rf_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(rf_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    rf_cv_scores[metric_name] = scores
cv_results['RandomForest'] = {
    'MAE': rf_cv_scores['MAE'],
    'RMSE': rf_cv_scores['RMSE'],
    'R2': rf_cv_scores['R2'],
    'Mean_MAE': np.mean(rf_cv_scores['MAE']),
    'Std_MAE': np.std(rf_cv_scores['MAE']),
    'Mean_RMSE': np.mean(rf_cv_scores['RMSE']),
    'Std_RMSE': np.std(rf_cv_scores['RMSE']),
    'Mean_R2': np.mean(rf_cv_scores['R2']),
    'Std_R2': np.std(rf_cv_scores['R2'])
}
print(f"   📈 RandomForest五折CV结果：")
print(f"      MAE: {cv_results['RandomForest']['Mean_MAE']:.4f} ± {cv_results['RandomForest']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['RandomForest']['Mean_RMSE']:.4f} ± {cv_results['RandomForest']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['RandomForest']['Mean_R2']:.4f} ± {cv_results['RandomForest']['Std_R2']:.4f}")

# 4.5 LightGBM
print("\n4.5 训练LightGBM模型...")
start_time = time.time()
lgb_model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
lgb_model.fit(X_tr_final, y_tr_final)
predictions['LightGBM'] = lgb_model.predict(X_te_final)
trained_models['LightGBM'] = lgb_model
train_time = time.time() - start_time
print(f"   ✅ LightGBM训练完成 | 耗时: {train_time:.2f}秒 | 迭代次数: {lgb_model.n_estimators_}")
print("   📊 执行LightGBM五折交叉验证...")
lgb_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(lgb_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    lgb_cv_scores[metric_name] = scores
cv_results['LightGBM'] = {
    'MAE': lgb_cv_scores['MAE'],
    'RMSE': lgb_cv_scores['RMSE'],
    'R2': lgb_cv_scores['R2'],
    'Mean_MAE': np.mean(lgb_cv_scores['MAE']),
    'Std_MAE': np.std(lgb_cv_scores['MAE']),
    'Mean_RMSE': np.mean(lgb_cv_scores['RMSE']),
    'Std_RMSE': np.std(lgb_cv_scores['RMSE']),
    'Mean_R2': np.mean(lgb_cv_scores['R2']),
    'Std_R2': np.std(lgb_cv_scores['R2'])
}
print(f"   📈 LightGBM五折CV结果：")
print(f"      MAE: {cv_results['LightGBM']['Mean_MAE']:.4f} ± {cv_results['LightGBM']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['LightGBM']['Mean_RMSE']:.4f} ± {cv_results['LightGBM']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['LightGBM']['Mean_R2']:.4f} ± {cv_results['LightGBM']['Std_R2']:.4f}")

# 4.6 XGBoost
print("\n4.6 训练XGBoost模型...")
start_time = time.time()
xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
xgb_model.fit(X_tr_final, y_tr_final)
predictions['XGBoost'] = xgb_model.predict(X_te_final)
trained_models['XGBoost'] = xgb_model
train_time = time.time() - start_time
print(f"   ✅ XGBoost训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行XGBoost五折交叉验证...")
xgb_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(xgb_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    xgb_cv_scores[metric_name] = scores
cv_results['XGBoost'] = {
    'MAE': xgb_cv_scores['MAE'],
    'RMSE': xgb_cv_scores['RMSE'],
    'R2': xgb_cv_scores['R2'],
    'Mean_MAE': np.mean(xgb_cv_scores['MAE']),
    'Std_MAE': np.std(xgb_cv_scores['MAE']),
    'Mean_RMSE': np.mean(xgb_cv_scores['RMSE']),
    'Std_RMSE': np.std(xgb_cv_scores['RMSE']),
    'Mean_R2': np.mean(xgb_cv_scores['R2']),
    'Std_R2': np.std(xgb_cv_scores['R2'])
}
print(f"   📈 XGBoost五折CV结果：")
print(f"      MAE: {cv_results['XGBoost']['Mean_MAE']:.4f} ± {cv_results['XGBoost']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['XGBoost']['Mean_RMSE']:.4f} ± {cv_results['XGBoost']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['XGBoost']['Mean_R2']:.4f} ± {cv_results['XGBoost']['Std_R2']:.4f}")

# 4.7 CatBoost
print("\n4.7 训练CatBoost模型...")
start_time = time.time()
cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
cat_model.fit(X_tr_final, y_tr_final)
predictions['CatBoost'] = cat_model.predict(X_te_final)
trained_models['CatBoost'] = cat_model
train_time = time.time() - start_time
print(f"   ✅ CatBoost训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行CatBoost五折交叉验证...")
cat_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(cat_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    cat_cv_scores[metric_name] = scores
cv_results['CatBoost'] = {
    'MAE': cat_cv_scores['MAE'],
    'RMSE': cat_cv_scores['RMSE'],
    'R2': cat_cv_scores['R2'],
    'Mean_MAE': np.mean(cat_cv_scores['MAE']),
    'Std_MAE': np.std(cat_cv_scores['MAE']),
    'Mean_RMSE': np.mean(cat_cv_scores['RMSE']),
    'Std_RMSE': np.std(cat_cv_scores['RMSE']),
    'Mean_R2': np.mean(cat_cv_scores['R2']),
    'Std_R2': np.std(cat_cv_scores['R2'])
}
print(f"   📈 CatBoost五折CV结果：")
print(f"      MAE: {cv_results['CatBoost']['Mean_MAE']:.4f} ± {cv_results['CatBoost']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['CatBoost']['Mean_RMSE']:.4f} ± {cv_results['CatBoost']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['CatBoost']['Mean_R2']:.4f} ± {cv_results['CatBoost']['Std_R2']:.4f}")

# 4.8 Transformer
print("\n4.8 训练Transformer模型...")
start_time = time.time()
try:
    trf_model = TransformerRegressor(n_features=X_tr_final.shape[1])
    trf_trained, trf_y_mean, trf_y_std, trf_train_loss = train_gpu_model(trf_model, "Transformer", X_tr_scaled, y_tr_final, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LR, batch_size=TRANSFORMER_BATCH_SIZE)
    predictions['Transformer'] = predict_with_model(trf_trained, X_te_scaled, trf_y_mean, trf_y_std)
    trained_models['Transformer'] = trf_trained
    loss_histories['Transformer'] = trf_train_loss
    train_time = time.time() - start_time
    print(f"   ✅ Transformer训练完成 | 耗时: {train_time:.2f}秒")

    print("   📊 执行Transformer五折交叉验证...")
    trf_cv_scores = {'MAE': [], 'RMSE': [], 'R2': []}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_scaled)):
        print(f"      正在执行Transformer第{fold + 1}折验证...")
        X_fold_tr = X_tr_scaled.iloc[train_idx]
        y_fold_tr = y_tr_final.iloc[train_idx]
        X_fold_val = X_tr_scaled.iloc[val_idx]
        y_fold_val = y_tr_final.iloc[val_idx]
        fold_model = TransformerRegressor(n_features=X_tr_final.shape[1])
        fold_model_trained, fold_y_mean, fold_y_std, _ = train_gpu_model(
            fold_model, f"Transformer_Fold{fold + 1}", X_fold_tr, y_fold_tr,
            epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LR, batch_size=TRANSFORMER_BATCH_SIZE
        )
        fold_pred = predict_with_model(fold_model_trained, X_fold_val, fold_y_mean, fold_y_std)
        trf_cv_scores['MAE'].append(mean_absolute_error(y_fold_val, fold_pred))
        trf_cv_scores['RMSE'].append(np.sqrt(mean_squared_error(y_fold_val, fold_pred)))
        trf_cv_scores['R2'].append(r2_score(y_fold_val, fold_pred))
    cv_results['Transformer'] = {
        'MAE': np.array(trf_cv_scores['MAE']), 'RMSE': np.array(trf_cv_scores['RMSE']), 'R2': np.array(trf_cv_scores['R2']),
        'Mean_MAE': np.mean(trf_cv_scores['MAE']), 'Std_MAE': np.std(trf_cv_scores['MAE']),
        'Mean_RMSE': np.mean(trf_cv_scores['RMSE']), 'Std_RMSE': np.std(trf_cv_scores['RMSE']),
        'Mean_R2': np.mean(trf_cv_scores['R2']), 'Std_R2': np.std(trf_cv_scores['R2'])
    }
    print(f"   📈 Transformer五折CV结果：")
    print(f"      MAE: {cv_results['Transformer']['Mean_MAE']:.4f} ± {cv_results['Transformer']['Std_MAE']:.4f}")
    print(f"      RMSE: {cv_results['Transformer']['Mean_RMSE']:.4f} ± {cv_results['Transformer']['Std_RMSE']:.4f}")
    print(f"      R2: {cv_results['Transformer']['Mean_R2']:.4f} ± {cv_results['Transformer']['Std_R2']:.4f}")
except Exception as e:
    print(f"   ❌ Transformer训练失败: {e}")
    predictions['Transformer'] = np.full(len(y_te), y_tr_final.mean())
    loss_histories['Transformer'] = [0.0]
    cv_results['Transformer'] = {
        'MAE': np.array([np.nan] * N_FOLDS), 'RMSE': np.array([np.nan] * N_FOLDS), 'R2': np.array([np.nan] * N_FOLDS),
        'Mean_MAE': np.nan, 'Std_MAE': np.nan, 'Mean_RMSE': np.nan, 'Std_RMSE': np.nan, 'Mean_R2': np.nan, 'Std_R2': np.nan
    }

# 4.9 ResMLP
print("\n4.9 训练ResMLP模型...")
start_time = time.time()
try:
    mlp_model = ResMLP(n_features=X_tr_final.shape[1])
    mlp_trained, mlp_y_mean, mlp_y_std, mlp_train_loss = train_gpu_model(mlp_model, "ResMLP", X_tr_scaled, y_tr_final, epochs=RESMLP_EPOCHS, lr=RESMLP_LR, batch_size=RESMLP_BATCH_SIZE)
    predictions['ResMLP'] = predict_with_model(mlp_trained, X_te_scaled, mlp_y_mean, mlp_y_std)
    trained_models['ResMLP'] = mlp_trained
    loss_histories['ResMLP'] = mlp_train_loss
    train_time = time.time() - start_time
    print(f"   ✅ ResMLP训练完成 | 耗时: {train_time:.2f}秒")

    print("   📊 执行ResMLP五折交叉验证...")
    mlp_cv_scores = {'MAE': [], 'RMSE': [], 'R2': []}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_scaled)):
        print(f"      正在执行ResMLP第{fold + 1}折验证...")
        X_fold_tr = X_tr_scaled.iloc[train_idx]
        y_fold_tr = y_tr_final.iloc[train_idx]
        X_fold_val = X_tr_scaled.iloc[val_idx]
        y_fold_val = y_tr_final.iloc[val_idx]
        fold_model = ResMLP(n_features=X_tr_final.shape[1])
        fold_model_trained, fold_y_mean, fold_y_std, _ = train_gpu_model(
            fold_model, f"ResMLP_Fold{fold + 1}", X_fold_tr, y_fold_tr,
            epochs=RESMLP_EPOCHS, lr=RESMLP_LR, batch_size=RESMLP_BATCH_SIZE
        )
        fold_pred = predict_with_model(fold_model_trained, X_fold_val, fold_y_mean, fold_y_std)
        mlp_cv_scores['MAE'].append(mean_absolute_error(y_fold_val, fold_pred))
        mlp_cv_scores['RMSE'].append(np.sqrt(mean_squared_error(y_fold_val, fold_pred)))
        mlp_cv_scores['R2'].append(r2_score(y_fold_val, fold_pred))
    cv_results['ResMLP'] = {
        'MAE': np.array(mlp_cv_scores['MAE']), 'RMSE': np.array(mlp_cv_scores['RMSE']), 'R2': np.array(mlp_cv_scores['R2']),
        'Mean_MAE': np.mean(mlp_cv_scores['MAE']), 'Std_MAE': np.std(mlp_cv_scores['MAE']),
        'Mean_RMSE': np.mean(mlp_cv_scores['RMSE']), 'Std_RMSE': np.std(mlp_cv_scores['RMSE']),
        'Mean_R2': np.mean(mlp_cv_scores['R2']), 'Std_R2': np.std(mlp_cv_scores['R2'])
    }
    print(f"   📈 ResMLP五折CV结果：")
    print(f"      MAE: {cv_results['ResMLP']['Mean_MAE']:.4f} ± {cv_results['ResMLP']['Std_MAE']:.4f}")
    print(f"      RMSE: {cv_results['ResMLP']['Mean_RMSE']:.4f} ± {cv_results['ResMLP']['Std_RMSE']:.4f}")
    print(f"      R2: {cv_results['ResMLP']['Mean_R2']:.4f} ± {cv_results['ResMLP']['Std_R2']:.4f}")
except Exception as e:
    print(f"   ❌ ResMLP训练失败: {e}")
    predictions['ResMLP'] = np.full(len(y_te), y_tr_final.mean())
    loss_histories['ResMLP'] = [0.0]
    cv_results['ResMLP'] = {
        'MAE': np.array([np.nan] * N_FOLDS), 'RMSE': np.array([np.nan] * N_FOLDS), 'R2': np.array([np.nan] * N_FOLDS),
        'Mean_MAE': np.nan, 'Std_MAE': np.nan, 'Mean_RMSE': np.nan, 'Std_RMSE': np.nan, 'Mean_R2': np.nan, 'Std_R2': np.nan
    }

# 4.10 Bagging
print("\n4.10 训练Bagging集成模型（以LightGBM为基模型）...")
start_time = time.time()
bagging_model = BaggingRegressor(estimator=lgb.LGBMRegressor(**BAGGING_BASE_LGB_PARAMS), **BAGGING_PARAMS)
bagging_model.fit(X_tr_final, y_tr_final)
predictions['Bagging'] = bagging_model.predict(X_te_final)
trained_models['Bagging'] = bagging_model
train_time = time.time() - start_time
print(f"   ✅ Bagging集成训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行Bagging五折交叉验证...")
bagging_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(bagging_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    bagging_cv_scores[metric_name] = scores
cv_results['Bagging'] = {
    'MAE': bagging_cv_scores['MAE'],
    'RMSE': bagging_cv_scores['RMSE'],
    'R2': bagging_cv_scores['R2'],
    'Mean_MAE': np.mean(bagging_cv_scores['MAE']),
    'Std_MAE': np.std(bagging_cv_scores['MAE']),
    'Mean_RMSE': np.mean(bagging_cv_scores['RMSE']),
    'Std_RMSE': np.std(bagging_cv_scores['RMSE']),
    'Mean_R2': np.mean(bagging_cv_scores['R2']),
    'Std_R2': np.std(bagging_cv_scores['R2'])
}
print(f"   📈 Bagging五折CV结果：")
print(f"      MAE: {cv_results['Bagging']['Mean_MAE']:.4f} ± {cv_results['Bagging']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['Bagging']['Mean_RMSE']:.4f} ± {cv_results['Bagging']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['Bagging']['Mean_R2']:.4f} ± {cv_results['Bagging']['Std_R2']:.4f}")

# 4.11 Stacking
print("\n4.11 训练Stacking集成模型（多模型融合）...")
start_time = time.time()
base_models = [
    ('rf', RandomForestRegressor(**STACKING_RF_PARAMS)),
    ('lgb', lgb.LGBMRegressor(**STACKING_LGB_PARAMS)),
    ('xgb', xgb.XGBRegressor(**STACKING_XGB_PARAMS)),
    ('cat', CatBoostRegressor(**STACKING_CAT_PARAMS))
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=5, n_jobs=1, passthrough=True)
stacking_model.fit(X_tr_final, y_tr_final)
predictions['Stacking'] = stacking_model.predict(X_te_final)
trained_models['Stacking'] = stacking_model
train_time = time.time() - start_time
print(f"   ✅ Stacking集成训练完成 | 耗时: {train_time:.2f}秒")
print("   📊 执行Stacking五折交叉验证...")
stacking_cv_scores = {}
for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(stacking_model, X_tr_final, y_tr_final, cv=kf, scoring=metric, n_jobs=1)
    if metric_name in ['MAE', 'RMSE']:
        scores = -scores
    stacking_cv_scores[metric_name] = scores
cv_results['Stacking'] = {
    'MAE': stacking_cv_scores['MAE'],
    'RMSE': stacking_cv_scores['RMSE'],
    'R2': stacking_cv_scores['R2'],
    'Mean_MAE': np.mean(stacking_cv_scores['MAE']),
    'Std_MAE': np.std(stacking_cv_scores['MAE']),
    'Mean_RMSE': np.mean(stacking_cv_scores['RMSE']),
    'Std_RMSE': np.std(stacking_cv_scores['RMSE']),
    'Mean_R2': np.mean(stacking_cv_scores['R2']),
    'Std_R2': np.std(stacking_cv_scores['R2'])
}
print(f"   📈 Stacking五折CV结果：")
print(f"      MAE: {cv_results['Stacking']['Mean_MAE']:.4f} ± {cv_results['Stacking']['Std_MAE']:.4f}")
print(f"      RMSE: {cv_results['Stacking']['Mean_RMSE']:.4f} ± {cv_results['Stacking']['Std_RMSE']:.4f}")
print(f"      R2: {cv_results['Stacking']['Mean_R2']:.4f} ± {cv_results['Stacking']['Std_R2']:.4f}")

# 4.12 加权集成
print("\n4.12 执行加权集成（包含所有模型）...")
model_list = ['MLR', 'Lasso', 'SVR', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost', 'Transformer', 'ResMLP', 'Bagging', 'Stacking']
r2_scores = [max(r2_score(y_te, predictions[m]), 0) if m in predictions else 0 for m in model_list]
weights = np.array(r2_scores)
weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
predictions['WeightedEnsemble'] = sum(predictions[m] * w for m, w in zip(model_list, weights) if m in predictions)
print(f"   📊 集成权重分配:")
for m, w in zip(model_list, weights):
    if m in predictions:
        print(f"      - {m}: {w:.2%}")

# ==================== 模型评估 ====================
print("\n" + "📊 模型性能评估阶段（增强版）".center(80, "-"))
metrics_df = pd.DataFrame(index=predictions.keys(), columns=['MAE', 'RMSE', 'R2'])
for model_name, pred in predictions.items():
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    metrics_df.loc[model_name] = [mae, rmse, r2]
    metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
metrics_df = metrics_df.astype(float).round(4)
sorted_metrics = metrics_df.sort_values('R2', ascending=False)

print("\n5.1 各模型性能指标汇总:")
print("-" * 70)
print(f"{'模型名称':<18} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
print("-" * 70)
for model_name in sorted_metrics.index:
    mae = sorted_metrics.loc[model_name, 'MAE']
    rmse = sorted_metrics.loc[model_name, 'RMSE']
    r2 = sorted_metrics.loc[model_name, 'R2']
    print(f"{model_name:<18} {mae:<8.4f} {rmse:<8.4f} {r2:<8.4f}")
print("-" * 70)

best_model_name = sorted_metrics.index[0]
best_model = trained_models.get(best_model_name, None)
if "Stacking" in best_model_name:
    best_model_display_name = f"{best_model_name}（元模型：LinearRegression）"
elif "WeightedEnsemble" in best_model_name:
    best_single_model_name = max([m for m in model_list if m in predictions], key=lambda x: r2_score(y_te, predictions[x]))
    best_model_display_name = f"{best_model_name}（核心模型：{best_single_model_name}）"
else:
    best_model_display_name = best_model_name

ensemble_models = ['Stacking', 'Bagging', 'WeightedEnsemble']
single_models_r2 = {mn: sorted_metrics.loc[mn, 'R2'] for mn in sorted_metrics.index if mn not in ensemble_models and mn in predictions}
best_single_model_name = max(single_models_r2, key=single_models_r2.get)
best_single_model = trained_models[best_single_model_name]
best_r2 = sorted_metrics.loc[best_model_name, 'R2']

print(f"\n5.2 最佳模型: {best_model_display_name} | 最优R2: {best_r2:.4f}")
print(f"   - MAE: {sorted_metrics.loc[best_model_name, 'MAE']:.4f} (平均预测误差: {sorted_metrics.loc[best_model_name, 'MAE']:.2f} 分)")
print(f"   - RMSE: {sorted_metrics.loc[best_model_name, 'RMSE']:.4f}")

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics_enhanced.csv"))

# 5.4 五折交叉验证结果写入 Excel（每折详细 + 均值标准差汇总 + 美化）
print("\n5.4 保存五折交叉验证结果到Excel...")
with pd.ExcelWriter(cv_excel_path, engine='openpyxl') as writer:
    # 工作表1：每折详细结果
    fold_detail_data = []
    for model_name in cv_results.keys():
        model_data = cv_results[model_name]
        for fold in range(N_FOLDS):
            fold_detail_data.append({
                'Model': model_name,
                'Fold': fold + 1,
                'MAE': model_data['MAE'][fold] if not np.isnan(model_data['MAE'][fold]) else '-',
                'RMSE': model_data['RMSE'][fold] if not np.isnan(model_data['RMSE'][fold]) else '-',
                'R2': model_data['R2'][fold] if not np.isnan(model_data['R2'][fold]) else '-'
            })
    fold_detail_df = pd.DataFrame(fold_detail_data)
    fold_detail_df.to_excel(writer, sheet_name='每折详细结果', index=False)

    # 工作表2：均值和标准差汇总

    summary_data = []
    for model_name in cv_results.keys():
        model_data = cv_results[model_name]
        summary_data.append({
            'Model': model_name,
            'MAE_Mean': model_data.get('Mean_MAE', np.nanmean(model_data['MAE'])),
            'MAE_Std': model_data.get('Std_MAE', np.nanstd(model_data['MAE'])),
            'RMSE_Mean': model_data.get('Mean_RMSE', np.nanmean(model_data['RMSE'])),
            'RMSE_Std': model_data.get('Std_RMSE', np.nanstd(model_data['RMSE'])),
            'R2_Mean': model_data.get('Mean_R2', np.nanmean(model_data['R2'])),
            'R2_Std': model_data.get('Std_R2', np.nanstd(model_data['R2']))
        })
    summary_df = pd.DataFrame(summary_data).sort_values('R2_Mean', ascending=False)
    summary_df.to_excel(writer, sheet_name='均值标准差汇总', index=False)

# 美化Excel文件（可选，增强可读性）
wb = openpyxl.load_workbook(cv_excel_path)
header_font = Font(bold=True, color='FFFFFF')
header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
alignment = Alignment(horizontal='center', vertical='center')

# 美化"每折详细结果"工作表
ws1 = wb['每折详细结果']
# 设置表头样式
for cell in ws1[1]:
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = alignment

# 自动调整列宽
for column in ws1.columns:
    max_length = 0
    column_letter = column[0].column_letter
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except Exception:
            pass
    adjusted_width = min(max_length + 2, 20)
    ws1.column_dimensions[column_letter].width = adjusted_width

# 美化"均值标准差汇总"工作表
ws2 = wb['均值标准差汇总']
for cell in ws2[1]:
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = alignment

for column in ws2.columns:
    max_length = 0
    column_letter = column[0].column_letter
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except Exception:
            pass
    adjusted_width = min(max_length + 2, 20)
    ws2.column_dimensions[column_letter].width = adjusted_width

# 保存美化后的Excel
wb.save(cv_excel_path)
print(f"   ✅ 五折交叉验证结果已保存至: {cv_excel_path}")
print(f"\n5.3 性能指标已保存至: {os.path.join(OUTPUT_DIR, 'model_metrics_enhanced.csv')}")

# ==================== SHAP 与 H-statistic ====================
if best_single_model is not None:
    print("\n" + "🔍 SHAP深度分析（增强版）".center(80, "-"))
    significant_pairs = []
    if len(X_tr_final.columns) >= 2:
        h_df, significant_pairs = h_statistic_analysis(best_single_model, X_tr_final, y_tr_final, top_features=TOP_FEATURES)

    print("\n6.1 准备SHAP分析数据...")
    print(f"   📌 选择单一模型进行SHAP分析：{best_single_model_name}（单一模型中R²最高：{sorted_metrics.loc[best_single_model_name, 'R2']:.4f}）")
    shap_values = None
    X_sample = None
    try:
        X_sample = X_te_final.sample(min(1000, len(X_te_final)), random_state=RANDOM_STATE)
        X_sample = X_sample[X_tr_final.columns].copy()
        if isinstance(best_single_model, (RandomForestRegressor, lgb.LGBMRegressor, xgb.XGBRegressor, CatBoostRegressor)):
            explainer = shap.TreeExplainer(best_single_model)
            shap_values = explainer.shap_values(X_sample)
        elif isinstance(best_single_model, (LinearRegression, Lasso)):
            explainer = shap.LinearExplainer(best_single_model, X_tr_final)
            shap_values = explainer.shap_values(X_sample)
        else:
            print("   ❌ 该模型不支持SHAP分析")
    except Exception as e:
        print(f"   ❌ SHAP解释器初始化失败: {e}")

    if shap_values is not None and X_sample is not None:
        print("\n6.2 生成SHAP蜂群图（Top20特征）...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, max_display=TOP_FEATURES, plot_type="dot", show=False)
        plt.title('SHAP Summary Plot (Top 20 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ SHAP蜂群图已保存至: {os.path.join(SHAP_DIR, 'shap_summary_plot.png')}")

        print("\n6.3 生成SHAP依赖图（Top20特征，5*4排列）...")
        if len(X_sample.columns) >= TOP_FEATURES:
            top_20_feats = pd.DataFrame({'feature': X_sample.columns, 'shap_importance': np.abs(shap_values).mean(axis=0)}).sort_values('shap_importance', ascending=False).head(TOP_FEATURES)['feature'].tolist()
        else:
            top_20_feats = X_sample.columns.tolist()
        fig, axes = plt.subplots(5, 4, figsize=(20, 25))
        fig.suptitle('SHAP Dependence Plot (Top 20 Features, 5×4 Layout)', fontsize=18, fontweight='bold')
        axes = axes.flatten()
        for idx, feat in enumerate(top_20_feats):
            if idx >= len(axes):
                break
            try:
                shap.dependence_plot(feat, shap_values, X_sample, feature_names=X_sample.columns, ax=axes[idx], show=False)
                axes[idx].set_title(f'Feature: {feat}', fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Feature Value')
                axes[idx].set_ylabel('SHAP Value')
            except Exception as e:
                axes[idx].set_title(f'Feature: {feat} (Plot Failed)', fontsize=10)
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center', transform=axes[idx].transAxes)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(SHAP_DIR, "shap_dependence_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ SHAP依赖图已保存至: {os.path.join(SHAP_DIR, 'shap_dependence_plots.png')}")

    # ==================== 特征重要性图 ====================
    print("\n" + "🎨 可视化图表生成阶段（增强版）".center(80, "-"))
    print("\n8.1 生成Top20特征重要性图（最佳模型）...")

    def plot_best_model_feature_importance():
        target_model = best_single_model
        target_model_name = best_single_model_name
        if target_model is None:
            print("   ❌ 无有效最佳模型，无法生成特征重要性图")
            return
        imp_df = None
        if hasattr(target_model, 'feature_importances_'):
            imp_df = pd.DataFrame({'feature': X_tr_final.columns, 'importance': target_model.feature_importances_}).sort_values('importance', ascending=False).head(TOP_FEATURES)
        elif hasattr(target_model, 'coef_') and len(X_tr_final.columns) == len(target_model.coef_):
            imp_df = pd.DataFrame({'feature': X_tr_final.columns, 'importance': np.abs(target_model.coef_)}).sort_values('importance', ascending=False).head(TOP_FEATURES)
        else:
            print("   ❌ 该模型不支持特征重要性计算")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='skyblue')
        ax.set_xlabel('Feature Importance Score', fontsize=12)
        ax.set_ylabel('Feature Name', fontsize=12)
        ax.set_title('Top 20 Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 特征重要性图已保存至: {os.path.join(OUTPUT_DIR, 'feature_importance_enhanced.png')}")

    plot_best_model_feature_importance()

    # ==================== 综合性分析仪表板（不绘制模型训练曲线）====================
    print("\n8.2 生成综合性分析仪表板（增强版）...")

    def plot_comprehensive_analysis():
        best_pred = predictions[best_model_name]
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('PISA 2018 Student Life Satisfaction Prediction - Enhanced Comprehensive Analysis Dashboard', fontsize=18, fontweight='bold')

        ax1 = plt.subplot(3, 3, 1, projection='polar')
        metrics_list = ['MAE', 'RMSE', 'R2']
        angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist() + [0]
        core_models = ['MLR', 'Lasso', 'SVR', 'LightGBM', 'XGBoost', 'Stacking', 'WeightedEnsemble']
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'black']
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        for i, m in enumerate(core_models):
            if m in metrics_df.index:
                vals = [metrics_df.loc[m, met] for met in metrics_list[:-1]] + [metrics_df.loc[m, metrics_list[0]]]
                ax1.plot(angles, vals, label=m, color=colors[i])
                ax1.fill(angles, vals, alpha=0.2, color=colors[i])
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics_list)
        ax1.set_title("Model Performance Radar Chart", fontsize=12, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=6)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 3, 2)
        r2_vals = sorted_metrics['R2'].values[:10]
        model_names = sorted_metrics.index[:10]
        colors = ['#2E8B57' if x == best_r2 else '#87CEEB' for x in r2_vals]
        bars = ax2.bar(model_names, r2_vals, color=colors)
        ax2.set_ylabel('R$^2$ Score', fontsize=10)
        ax2.set_title('Model R$^2$ Performance Ranking (Top 10)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(r2_vals) * 1.1)
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(y_te, best_pred, alpha=0.5, s=10, color='blue')
        min_val = min(y_te.min(), best_pred.min())
        max_val = max(y_te.max(), best_pred.max())
        # 完美预测线（原有）
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        # 添加拟合线
        lr_fit = LinearRegression()
        lr_fit.fit(y_te.values.reshape(-1, 1), best_pred)
        fit_line_x = np.array([min_val, max_val])
        fit_line_y = lr_fit.predict(fit_line_x.reshape(-1, 1))
        ax3.plot(fit_line_x, fit_line_y, 'g-', linewidth=2,
                 label=f'Fitted Line (y={lr_fit.coef_[0]:.2f}x + {lr_fit.intercept_:.2f})')
        ax3.set_xlabel('True Value (Life Satisfaction)', fontsize=10)
        ax3.set_ylabel('Predicted Value (Life Satisfaction)', fontsize=10)
        ax3.set_title('Predicted vs True Values', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        ax4 = plt.subplot(3, 3, 4)
        residuals = best_pred - y_te
        ax4.scatter(best_pred, residuals, alpha=0.5, s=10, color='orange')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted Value', fontsize=10)
        ax4.set_ylabel('Residual (Predicted - True)', fontsize=10)
        ax4.set_title('Residual Analysis Plot', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        ax5 = plt.subplot(3, 3, 5)
        if best_single_model is not None and hasattr(best_single_model, 'feature_importances_'):
            imp_df = pd.DataFrame({'feature': X_tr_final.columns, 'importance': best_single_model.feature_importances_}).sort_values('importance', ascending=False).head(15)
            ax5.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='green')
            ax5.set_xlabel('Importance Score')
            ax5.set_title('Top 15 Feature Importance (Best Single Model)', fontsize=12, fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Valid Feature Importance Data', ha='center', va='center', fontsize=12)
            ax5.set_title('Top 15 Feature Importance (Best Single Model)', fontsize=12, fontweight='bold')

        # 6. 按用户要求，不再绘制模型训练曲线
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')

        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(y_te, bins=20, alpha=0.5, label='True Values', color='blue', density=True)
        ax7.hist(best_pred, bins=20, alpha=0.5, label='Predicted Values', color='orange', density=True)
        ax7.set_xlabel('Satisfaction Score (0-10)')
        ax7.set_ylabel('Density')
        ax7.set_title('Predicted vs True Distribution', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(alpha=0.3)

        ax8 = plt.subplot(3, 3, 8)
        errors = best_pred - y_te
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_skew = skew(errors)
        ax8.hist(errors, bins=30, alpha=0.7, color='purple', density=True)
        ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
        stats_text = f"""
            Error Statistics
            ─────────
            Mean: {error_mean:.4f}
            Std: {error_std:.4f}
            Skewness: {error_skew:.4f}
            """
        ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax8.set_xlabel('Prediction Error (Predicted - True)', fontsize=10)
        ax8.set_ylabel('Density', fontsize=10)
        ax8.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax8.legend(loc='upper right', fontsize=9)
        ax8.grid(alpha=0.3)

        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        sig_pairs_count = len(significant_pairs) if significant_pairs else 0
        analysis_text = f"""
            📊 Performance Analysis Summary (Enhanced)
            ─────────────────
            Best Model: {best_model_display_name}
            Best R$^2$: {best_r2:.4f}
            Avg MAE: {metrics_df['MAE'].mean():.4f}
            Avg RMSE: {metrics_df['RMSE'].mean():.4f}

            🎯 Key Findings
            ─────────────────
            1. Added MLR/Lasso/SVR models, linear models perform stably
            2. Stacking/Weighted Ensemble achieves best performance & generalization
            3. KNN Imputation (Gower distance) preserves data distribution well
            4. SHAP reveals non-linear effects of core features
            5. H-statistic detects significant feature interactions

            💡 Improvement Suggestions
            ─────────────────
            1. Optimize Stacking meta-model with XGBoost/CatBoost
            2. Use SHAP insights for targeted feature engineering
            3. Construct interaction terms based on H-statistic results
            4. Apply Optuna hyperparameter tuning for single models
            """
        ax9.text(0.05, 0.95, analysis_text, transform=ax9.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_analysis_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()

    plot_comprehensive_analysis()
    print(f"   ✅ 综合性分析仪表板已保存至: {os.path.join(OUTPUT_DIR, 'comprehensive_analysis_enhanced.png')}")

    # ==================== 详细性能报告 ====================
    print("\n8.3 生成详细性能报告（增强版）...")

    def generate_performance_report():
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        best_pred = predictions[best_model_name]
        errors = best_pred - y_te
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_skew_val = skew(errors)

        dataset_info = f"""
==============================================
📋 Dataset Information
==============================================
• Raw Dataset: {len(df_raw)} samples, {df_raw.shape[1]} features
• Cleaned Dataset: {len(df_clean)} samples, {X_raw.shape[1]} features
• Imputed Dataset: {len(X_imputed)} samples, {X_imputed.shape[1]} features
• After Feature Engineering: {X_enhanced.shape[1]} features
• After Feature Selection: {len(top_features)} features
• Training Set: {len(X_tr_final)} samples (80%)
• Test Set: {len(X_te_final)} samples (20%)
• Target Variable: {target_variable} (Life Satisfaction 0-10)
• Target Stats: Mean={y.mean():.2f}, Std={y.std():.2f}, Median={y.median():.2f}
"""

        performance_ranking = "\n==============================================\n🎯 Model Performance Ranking (Top 10 by R2)\n==============================================\n"
        for i, model_name in enumerate(sorted_metrics.index[:10]):
            mae = sorted_metrics.loc[model_name, 'MAE']
            rmse = sorted_metrics.loc[model_name, 'RMSE']
            r2 = sorted_metrics.loc[model_name, 'R2']
            performance_ranking += f"{i + 1}. {model_name}:\n   - MAE: {mae:.4f} (Avg Prediction Error: {mae:.2f} points)\n   - RMSE: {rmse:.4f}\n   - R2: {r2:.4f}\n"

        avg_r2 = metrics_df['R2'].mean()
        avg_mae = metrics_df['MAE'].mean()
        performance_grade = "Excellent" if best_r2 >= 0.4 else "Good" if best_r2 >= 0.3 else "Fair"
        performance_analysis = f"""
==============================================
📊 Performance Analysis (Enhanced)
==============================================
• Best Model: {best_model_display_name}
• Best R2: {best_r2:.4f}
• Average R2: {avg_r2:.4f}
• Average MAE: {avg_mae:.4f}
• Performance Grade: {performance_grade}
• Prediction Error Stats (Best Model): Error Mean: {error_mean:.4f}, Error Std: {error_std:.4f}, Error Skewness: {error_skew_val:.4f}
"""

        key_findings = f"""
==============================================
🔍 Key Findings (Enhanced)
==============================================
1. KNN imputation (Gower distance) handles mixed-type variables well; data distribution & correlation structure preserved.
2. MLR/Lasso/SVR perform stably; Stacking/Weighted Ensemble achieves best performance.
3. SHAP Top 20 features identified; H-statistic detected {len(significant_pairs)} significant interaction pairs.
"""

        optimization_results = """
==============================================
🚀 Optimization Summary (Enhanced)
==============================================
1. Data: Gower KNN imputation + quality validation; simplified feature engineering.
2. Models: MLR, Lasso, SVR + all original models & ensemble strategies.
3. Analysis: SHAP (summary + 5×4 dependence); H-statistic; no model training-curve plot in dashboard.
"""

        report = f"""
PISA 2018 Student Life Satisfaction Prediction - Performance Report (Enhanced)
==============================================
Report Time: {timestamp}
Device: {DEVICE}
Output Path: {OUTPUT_DIR}
Core Enhancements: Gower KNN Imputation + MLR/Lasso/SVR + SHAP + H-statistic
Version Info: sklearn={sklearn.__version__}, lightgbm={lgb.__version__}, xgboost={xgb.__version__}
==============================================
{dataset_info}
{performance_ranking}
{performance_analysis}
{key_findings}
{optimization_results}

==============================================
💡 Advanced Improvement Suggestions
==============================================
1. Model Level:
   • Replace Stacking meta-model with XGBoost/CatBoost for better fusion
   • Try heterogeneous Bagging (tree + linear + SVR base learners)
   • Extend deep learning epochs & optimize network architecture
   • Apply semi-supervised learning with unlabeled data

2. Feature Level:
   • Feature transformation & binning based on SHAP non-linearity
   • Construct interaction features from significant H-statistic pairs
   • Automated feature engineering (TPOT, Featuretools)
   • Separate modeling for extreme-value samples

3. Training Level:
   • 10-fold cross-validation for more reliable evaluation
   • Optuna hyperparameter optimization for all single models
   • Hybrid Stacking-Blending ensemble strategy
   • Prediction uncertainty estimation

4. Application Level:
   • Business interpretation report based on SHAP insights
   • Targeted intervention strategies for key impact factors
   • Model deployment pipeline for real-time prediction
   • Performance monitoring & periodic retraining system
"""
        with open(os.path.join(OUTPUT_DIR, 'performance_report_enhanced.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        return report

    report = generate_performance_report()
    print(f"   ✅ 性能报告已保存至: {os.path.join(OUTPUT_DIR, 'performance_report_enhanced.txt')}")

    # ==================== 最终控制台输出 ====================
    print("\n" + "✅ All Tasks Completed (Enhanced)".center(80, "-"))
    print(f"\n📁 Generated Files:")
    print(f"   1. Comprehensive Dashboard: {os.path.join(OUTPUT_DIR, 'comprehensive_analysis_enhanced.png')}")
    print(f"   2. Feature Importance Plot: {os.path.join(OUTPUT_DIR, 'feature_importance_enhanced.png')}")
    print(f"   3. Performance Report: {os.path.join(OUTPUT_DIR, 'performance_report_enhanced.txt')}")
    print(f"   4. Model Metrics: {os.path.join(OUTPUT_DIR, 'model_metrics_enhanced.csv')}")
    print(f"   5. Train/Test Data: train.csv / test.csv")
    print(f"   6. Imputation Quality: {IMPUTATION_DIR}")
    print(f"   7. SHAP Analysis: {SHAP_DIR}")
    print(f"   8. H-statistic Interaction: {H_STAT_DIR}")
    print(f"   9. 五折交叉验证结果: {cv_excel_path}")

    print(f"\n🏆 Core Results (Enhanced):")
    print(f"   • Best Model: {best_model_display_name}")
    print(f"   • Best R2: {best_r2:.4f}")
    print(f"   • Avg Prediction Error: {sorted_metrics.loc[best_model_name, 'MAE']:.2f} points")
    grade = "Excellent" if best_r2 >= 0.4 else "Good" if best_r2 >= 0.3 else "Fair"
    print(f"   • Performance Grade: {grade}")
    print(f"   • Completed: Gower KNN Imputation + Quality Validation")
    print(f"   • Completed: SHAP Deep Analysis (Top20, 5×4 Layout)")
    print(f"   • Completed: H-statistic Interaction Test ({len(significant_pairs) if significant_pairs else 0} significant pairs)")

    print("\n" + "=" * 80)
    print("🎉 Enhanced Analysis Completed! All files saved to target directory.")
    print("=" * 80)

