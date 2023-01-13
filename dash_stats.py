import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.stats.proportion import proportion_confint


def general_data_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        "n": str(len(df)),
        "n positives": str(int(df["y_true"].sum())),
        "n negatives": str(int(len(df) - df["y_true"].sum())),
        "prevalence": str(round(df["y_true"].sum() / len(df), 3)),
    }
    return pd.DataFrame([stats]).T


def general_performance(
    y_true: np.ndarray, y_score: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    general_metrics = {}
    general_metrics["AUC"] = round(metrics.auc(fpr, tpr), 3)
    general_metrics["pAUC (sp 0.9-1)"] = round(metrics.roc_auc_score(y_true, y_score, max_fpr=0.1), 3)
    cutoff_index = np.argmin(abs(tpr - (1 - fpr)))
    specifcity = (1 - fpr)[cutoff_index]
    general_metrics["(sp, sn) at sp=sn"] = str((round(specifcity, 3), round(tpr[cutoff_index], 3)))
    general_metrics["threshold at sp=sn"] = str(round(thresholds[cutoff_index], 3))
    precisions, recalls, pr_thresholds = metrics.precision_recall_curve(y_true, y_score)
    general_metrics["AUC_PR"] = str(round(metrics.auc(recalls, precisions), 3))
    return pd.DataFrame([general_metrics]).astype(str).T


def specificity_at_sensitivity(sensitivity, y_true, y_score):
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_score)
    index = np.where(tpr >= sensitivity)[0][0]
    specificity = 1 - fpr[index]
    tn = np.logical_and(y_true == 0, y_score <= roc_thresholds[index]).sum()
    specificity_ci = proportion_confint(count=tn, nobs=int((y_true == 0).sum()), method="agresti_coull")
    return (specificity, specificity_ci)


def performance_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, specificity
) -> pd.DataFrame:
    performance = {}
    index = np.where(1 - fpr >= specificity)[0][-1]
    sensitivity = tpr[index]
    tp = np.logical_and(y_true == 1, y_score > thresholds[index]).sum()
    sensitivity_ci = proportion_confint(count=tp, nobs=int(y_true.sum()), method="agresti_coull")
    sensitivity_ci = str((round(sensitivity_ci[0], 3), round(sensitivity_ci[1], 3)))
    performance["senseitivity/recall, (CI)"] = f"{str(round(sensitivity, 3))}, {sensitivity_ci}"
    specificity, specificity_ci = specificity_at_sensitivity(sensitivity, y_true, y_score)
    performance[
        "specificity, (CI)"
    ] = f"{round(specificity, 3)}, ({round(specificity_ci[0], 3)}, {round(specificity_ci[1], 3)})"
    performance["threshold"] = str(thresholds[index])
    return pd.DataFrame([performance]).T


def get_sn_sp_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    d = dict.fromkeys(["sensitivity", "specificity"], np.nan)
    if len(y_true) == y_true.sum():
        # all true
        tp = (y_score >= threshold).sum()
        fn = (y_score < threshold).sum()
        sensitivity = round(tp / (tp + fn), 3)
        sn_ci = proportion_confint(count=tp, nobs=int(y_true.sum()), method="agresti_coull")
        d = {"sensitivity": f"{sensitivity}, ({round(sn_ci[0], 3)}, {round(sn_ci[1], 3)})"}
    elif y_true.sum() == 0:
        # all false
        tn = (y_score <= threshold).sum()
        fp = (y_score > threshold).sum()
        specificity = round(tn / (tn + fp), 3)
        sp_ci = proportion_confint(count=tn, nobs=int((y_true == 0).sum()), method="agresti_coull")
        d = {"specificity": f"{specificity}, ({round(sp_ci[0], 3)}, {round(sp_ci[1], 3)})"}
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        threshold_index = np.argmax(thresholds <= threshold)
        sensitivity = tpr[threshold_index]
        specificity = 1 - fpr[threshold_index]
        tp = np.logical_and(y_true == 1, y_score > thresholds[threshold_index]).sum()
        tn = np.logical_and(y_true == 0, y_score <= thresholds[threshold_index]).sum()
        sn_ci = proportion_confint(count=tp, nobs=int(y_true.sum()), method="agresti_coull")
        sp_ci = proportion_confint(count=tn, nobs=int((y_true == 0).sum()), method="agresti_coull")
        d = {
            "sensitivity": f"{round(sensitivity, 3)}, ({round(sn_ci[0], 3)}, {round(sn_ci[1], 3)})",
            "specificity": f"{round(specificity, 3)}, ({round(sp_ci[0], 3)}, {round(sp_ci[1], 3)})",
        }
    return pd.DataFrame([d]).T
