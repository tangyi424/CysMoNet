# File: autogluon_runner.py
import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from utils import compute_metrics
from xgboost import XGBClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_and_evaluate_ag(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, outdir: str, label_col: str='label', time_limit: int=600, seed: int=42):
    """Train AutoGluon TabularPredictor on flattened features.
    X_*: numpy arrays; will be converted into DataFrames with columns f0..f{D-1}
    Returns metrics dict on test set.
    """
    os.makedirs(outdir, exist_ok=True)
    def to_df(X, y=None):
        cols = [f'f{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        if y is not None:
            df[label_col] = y
        return df

    train_df = to_df(X_train, y_train)
    val_df = to_df(X_val, y_val)
    # combine train+val for autogluon training; but pass validation as holdout via tuning? We'll concatenate and let inner CV handle
    train_df_full = pd.concat([train_df, val_df], ignore_index=True)
    test_df = to_df(X_test, None)

    predictor = TabularPredictor(label=label_col, path=outdir, problem_type='binary', eval_metric='roc_auc').fit(  # 'roc_auc'
        train_data=train_df_full,
        presets= 'best_quality' ,   #'best_quality'
        verbosity=2,  # 打印更多信息
    )

    # predict
    y_pred = predictor.predict(test_df)
    y_probs = predictor.predict_proba(test_df)[1]  # probability for positive class

    metrics = compute_metrics(y_test, y_pred.values, y_scores=y_probs.values)
    # print and return
    print('AutoGluon metrics:', metrics)
    return metrics, predictor


# def train_and_evaluate_ag(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     outdir: str,
#     label_col: str = 'label',
#     time_limit: int = 600,
#     seed: int = 42
# ):
#     """
#     Train an XGBoost classifier on flattened features.
#     X_*: numpy arrays; will be converted into DataFrames with columns f0..f{D-1}
#     Returns metrics dict on test set.
#     """
#     os.makedirs(outdir, exist_ok=True)
#
#     def to_df(X, y=None):
#         cols = [f"f{i}" for i in range(X.shape[1])]
#         df = pd.DataFrame(X, columns=cols)
#         if y is not None:
#             df[label_col] = y
#         return df
#
#     # === Prepare Data ===
#     train_df = to_df(X_train, y_train)
#     val_df = to_df(X_val, y_val)
#     test_df = to_df(X_test)
#
#     # Combine train + val for final training
#     train_full = pd.concat([train_df, val_df], ignore_index=True)
#     X_full = train_full.drop(columns=[label_col])
#     y_full = train_full[label_col]
#
#     # ⚠️ 关键修正：验证集也要转为 DataFrame
#     X_val_df = val_df.drop(columns=[label_col])
#     y_val_df = val_df[label_col]
#
#     # === Train XGBoost ===
#     print("Training XGBoost classifier...")
#
#     model = XGBClassifier(
#         n_estimators=500,
#         learning_rate=0.05,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='binary:logistic',
#         random_state=seed,
#         n_jobs=-1,
#         eval_metric='auc',
#         use_label_encoder=False
#     )
#
#     model.fit(
#         X_full,
#         y_full,
#         eval_set=[(X_val_df, y_val_df)],  # ✅ 修正：传 DataFrame 而不是 numpy
#         verbose=True,
#     )
#
#     # === Predict ===
#     y_pred = model.predict(X_test)
#     y_probs = model.predict_proba(X_test)[:, 1]
#
#     # === Evaluate ===
#     metrics = compute_metrics(y_test, y_pred, y_scores=y_probs)
#     print("XGBoost metrics:", metrics)
#
#     # === Save model ===
#     model_path = os.path.join(outdir, "xgboost_model.pkl")
#     joblib.dump(model, model_path)
#     print(f"✅ XGBoost model saved to: {model_path}")
#
#     return metrics, model


# def train_and_evaluate_ag(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     outdir: str,
#     label_col: str = "label",
#     time_limit: int = 600,
#     seed: int = 42,
# ):
#     """
#     Train a Random Forest classifier on flattened features.
#     X_*: numpy arrays; will be converted into DataFrames with columns f0..f{D-1}
#     Returns (metrics, model)
#     """
#     os.makedirs(outdir, exist_ok=True)
#
#     # === 数据转换函数 ===
#     def to_df(X, y=None):
#         cols = [f"f{i}" for i in range(X.shape[1])]
#         df = pd.DataFrame(X, columns=cols)
#         if y is not None:
#             df[label_col] = y
#         return df
#
#     # === 准备数据 ===
#     train_df = to_df(X_train, y_train)
#     val_df = to_df(X_val, y_val)
#     test_df = to_df(X_test)
#
#     # 合并训练集与验证集用于最终训练
#     train_full = pd.concat([train_df, val_df], ignore_index=True)
#     X_full = train_full.drop(columns=[label_col])
#     y_full = train_full[label_col]
#
#     # === 训练随机森林 ===
#     print("Training Random Forest classifier...")
#
#     model = RandomForestClassifier(
#         n_estimators=500,       # 森林中树的数量
#         max_depth=None,         # 不限制深度
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features="sqrt",    # 随机选取特征子集
#         bootstrap=True,
#         n_jobs=-1,
#         random_state=seed,
#     )
#
#     model.fit(X_full, y_full)
#
#     # === 预测 ===
#     y_pred = model.predict(X_test)
#     if hasattr(model, "predict_proba"):
#         y_probs = model.predict_proba(X_test)[:, 1]
#     else:
#         y_probs = y_pred  # 若不支持 predict_proba（不太可能）
#
#     # === 计算指标 ===
#     metrics = compute_metrics(y_test, y_pred, y_scores=y_probs)
#     print("RandomForest metrics:", metrics)
#
#     # === 保存模型 ===
#     model_path = os.path.join(outdir, "rf_model.pkl")
#     joblib.dump(model, model_path)
#     print(f"✅ Random Forest model saved to: {model_path}")
#
#     return metrics, model

# def train_and_evaluate_ag(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     outdir: str,
#     label_col: str = "label",
#     time_limit: int = 600,
#     seed: int = 42,
# ):
#     """
#     Train an SVM classifier on flattened features.
#     X_*: numpy arrays; will be converted into DataFrames with columns f0..f{D-1}
#     Returns (metrics, model)
#     """
#     os.makedirs(outdir, exist_ok=True)
#
#     # === 数据转换函数 ===
#     def to_df(X, y=None):
#         cols = [f"f{i}" for i in range(X.shape[1])]
#         df = pd.DataFrame(X, columns=cols)
#         if y is not None:
#             df[label_col] = y
#         return df
#
#     # === 准备数据 ===
#     train_df = to_df(X_train, y_train)
#     val_df = to_df(X_val, y_val)
#     test_df = to_df(X_test)
#
#     # 合并训练集与验证集用于最终训练
#     train_full = pd.concat([train_df, val_df], ignore_index=True)
#     X_full = train_full.drop(columns=[label_col]).values
#     y_full = train_full[label_col].values
#     X_test_np = test_df.values
#
#     # === 创建 SVM 分类器 ===
#     print("Training SVM classifier...")
#
#     # 使用标准化 + RBF 核心 SVM
#     model = make_pipeline(
#         StandardScaler(),
#         SVC(
#             kernel="rbf",
#             C=1.0,
#             gamma="scale",
#             probability=True,  # 启用 predict_proba
#             random_state=seed,
#         ),
#     )
#
#     model.fit(X_full, y_full)
#
#     # === 预测 ===
#     y_pred = model.predict(X_test_np)
#     if hasattr(model, "predict_proba"):
#         y_probs = model.predict_proba(X_test_np)[:, 1]
#     else:
#         # 若不支持概率输出，则用决策函数 sigmoid 近似
#         y_scores = model.decision_function(X_test_np)
#         y_probs = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
#
#     # === 计算指标 ===
#     metrics = compute_metrics(y_test, y_pred, y_scores=y_probs)
#     print("SVM metrics:", metrics)
#
#     # === 保存模型 ===
#     model_path = os.path.join(outdir, "svm_model.pkl")
#     joblib.dump(model, model_path)
#     print(f"✅ SVM model saved to: {model_path}")
#
#     return metrics, model