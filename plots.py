import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics


def plot_labeled_probabilites(y_true: np.array, y_score: np.array):
    proba_df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    fig = px.histogram(
        proba_df,
        x="y_score",
        color="y_true",
        histnorm="probability",
        title="Score probability by label",
        width=420,
        height=450,
        nbins=20,
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white", overwrite=True, marker={"opacity": 0.7})
    fig.update_xaxes(range=[0.0, 1.0])
    fig.update_layout(barmode="overlay")
    return fig


def plot_auc_roc(fpr, tpr, name="ROC"):
    best_threshold_index = np.argmax(tpr - fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name))
    fig.update_layout(
        title="AUC-ROC",
        xaxis_title="FPR (1-specificity)",
        yaxis_title="TPR (sensitivity)",
        hovermode="x",
        width=420,
        height=450,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(fpr[best_threshold_index]),
            y=np.array(tpr[best_threshold_index]),
            marker=dict(size=15, color="red"),
            name="sn=sp",
        )
    )
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(range=[0, 1])
    return fig


def plot_statistics_by_threshold(
    y_true: np.array, y_score: np.array, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
):
    all_metrics = {}
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    all_metrics["thresholds"] = thresholds
    all_metrics["sensitivity"] = tpr
    all_metrics["specificity"] = 1 - fpr
    df = pd.DataFrame(all_metrics)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=df["sensitivity"].values, name="Sensitivity"))
    fig.add_trace(go.Scatter(x=thresholds, y=df["specificity"].values, name="Specificity"))
    fig.update_layout(
        title="sensitivity (recall) and specificity for different thresholds",
        xaxis_title="Threshold",
        hovermode="x",
        width=450,
        height=450,
    )
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(range=[0, 1])
    return fig


def plot_fn_error_analysis(df: pd.DataFrame, title: str):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=col))
    fig.update_layout(title=title, hovermode="x", width=1200)
    return fig
