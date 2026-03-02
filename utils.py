# File: utils.py
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, average_precision_score, confusion_matrix, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray=None, threshold_step: float = 0.005) -> Dict:
    """Compute ACC, SEN (recall), SPEC, MCC, AP (if scores provided).
    y_true: binary 0/1
    y_pred: binary 0/1
    y_scores: continuous scores for AP
    """
    # acc = float(accuracy_score(y_true, y_pred))
    # sen = float(recall_score(y_true, y_pred))
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    # mcc = float(matthews_corrcoef(y_true, y_pred))
    # pre = float(precision_score(y_true, y_pred))
    # auroc = float(roc_auc_score(y_true, y_scores))  # ROC-AUC
    # aupr = float(average_precision_score(y_true, y_scores))


    best_threshold = 0.1
    best_mcc = -1.0
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    for th in thresholds:
        y_pred_th = (y_scores >= th).astype(int)
        mcc_th = matthews_corrcoef(y_true, y_pred_th)
        if mcc_th > best_mcc:
            best_mcc = mcc_th
            best_threshold = th
            y_pred = y_pred_th  # 使用最佳阈值更新 y_pred

    acc = float(accuracy_score(y_true, y_pred))
    sen = float(recall_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mcc = float(matthews_corrcoef(y_true, y_pred))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    auroc = float(roc_auc_score(y_true, y_scores))
    aupr = float(average_precision_score(y_true, y_scores))



    return {
        "THRESH": best_threshold,
        "ACC": acc,
        "SEN": sen,
        "SPEC": spec,
        "MCC": mcc,
        "PRE": pre,
        "AUROC": auroc,
        "AUPR": aupr
    }


def flatten_embedding(emb: np.ndarray) -> np.ndarray:
    """Ensure embedding is 1D array
    """
    arr = np.asarray(emb)
    return arr.reshape(-1)
