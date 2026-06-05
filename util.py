import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import namedtuple

# 定义返回结果的数据结构
MetricsResult = namedtuple('MetricsResult',
                           ['ACC', 'AUC', 'EER', 'FPR', 'FNR', 'TPR', 'TNR', 'Precision', 'Recall', 'F1'])


def cal_metrics(labels, predictions, threshold=0.5):
    """
    计算二分类任务的各项评估指标

    Args:
        labels: 真实标签列表 (0或1)
        predictions: 预测概率列表 (正类的概率)
        threshold: 分类阈值

    Returns:
        MetricsResult: 包含各项指标的结果对象
    """

    # 转换为numpy数组
    labels = np.array(labels)
    predictions = np.array(predictions)

    # 确保标签是0和1
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be 0 or 1")

    # 确保预测概率在[0,1]范围内
    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError("Predictions must be probabilities in [0, 1]")

    # 计算基于阈值的预测结果
    binary_predictions = (predictions >= threshold).astype(int)

    # 计算准确率
    accuracy = np.mean(binary_predictions == labels)

    # 计算AUC
    try:
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        auc_score = metrics.auc(fpr, tpr)
    except ValueError:
        # 如果只有一种类别，AUC无法计算
        auc_score = 0.5

    # 计算EER (Equal Error Rate)
    try:
        eer = calculate_eer(labels, predictions)
    except:
        eer = 0.5

    # 计算混淆矩阵相关指标
    tn, fp, fn, tp = metrics.confusion_matrix(labels, binary_predictions).ravel()
    # print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')

    # 计算各项指标
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    tnr_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr_rate  # 同上
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return MetricsResult(
        ACC=accuracy,
        AUC=auc_score,
        EER=eer,
        FPR=fpr_rate,
        FNR=fnr_rate,
        TPR=tpr_rate,
        TNR=tnr_rate,
        Precision=precision,
        Recall=recall,
        F1=f1_score
    )


def calculate_eer(labels, predictions):
    """
    计算等错误率 (Equal Error Rate)

    Args:
        labels: 真实标签
        predictions: 预测概率

    Returns:
        float: EER值
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

    # 寻找FPR = FNR的点 (FNR = 1 - TPR)
    fnr = 1 - tpr

    # 使用插值找到FPR = FNR的点
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer
