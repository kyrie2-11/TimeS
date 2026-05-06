"""
异常检测评估指标工具
==================
提供点级与事件级指标，适配工业数据集（SWaT/SMD/MSL）。
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_pointwise_metrics(y_true_anomaly: np.ndarray, y_pred_anomaly: np.ndarray) -> dict:
    """点级指标，输入需为 anomaly=1, normal=0。"""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_anomaly,
        y_pred_anomaly,
        average='binary',
        zero_division=0,
    )
    acc = accuracy_score(y_true_anomaly, y_pred_anomaly)
    return {
        'point_precision': float(p),
        'point_recall': float(r),
        'point_f1': float(f1),
        'point_accuracy': float(acc),
    }


def point_adjust_predictions(y_true_anomaly: np.ndarray, y_pred_anomaly: np.ndarray) -> np.ndarray:
    """
    Point-Adjust: 若一个真实异常段中命中任一点，则整段预测为异常。
    """
    y_true = y_true_anomaly.astype(int)
    y_pred = y_pred_anomaly.astype(int).copy()

    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 0:
            i += 1
            continue

        j = i
        while j + 1 < n and y_true[j + 1] == 1:
            j += 1

        if np.any(y_pred[i:j + 1] == 1):
            y_pred[i:j + 1] = 1

        i = j + 1

    return y_pred


def compute_event_metrics(y_true_anomaly: np.ndarray, y_pred_anomaly: np.ndarray) -> dict:
    """事件级近似指标（基于 point-adjust 后的 P/R/F1）。"""
    pa_pred = point_adjust_predictions(y_true_anomaly, y_pred_anomaly)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_anomaly,
        pa_pred,
        average='binary',
        zero_division=0,
    )
    return {
        'event_precision': float(p),
        'event_recall': float(r),
        'event_f1': float(f1),
    }
