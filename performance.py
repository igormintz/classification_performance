import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics
from st_aggrid import AgGrid, DataReturnMode

from dash_stats import (
    general_data_stats,
    general_performance,
    get_sn_sp_at_threshold,
    performance_at_specificity,
)
from plots import plot_auc_roc, plot_labeled_probabilites, plot_statistics_by_threshold


def show_plots(y_true: np.array, y_score: np.array, fpr: np.array, tpr: np.array, thresholds: np.array):
    st.subheader("Plots")
    col4, col5, col6 = st.columns([1, 1, 1])
    col4.plotly_chart(plot_statistics_by_threshold(y_true, y_score, fpr, tpr, thresholds))
    col5.plotly_chart(plot_auc_roc(fpr, tpr))
    col6.plotly_chart(plot_labeled_probabilites(y_true, y_score))


def show_tables_with_statistics(
    df: pd.DataFrame,
    y_true: np.array,
    y_score: np.array,
    fpr: np.array,
    tpr: np.array,
    thresholds: np.array,
    key="defalut",
):
    # st.subheader(label)
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.subheader("Data")
    col1.dataframe(general_data_stats(df))
    col2.subheader("General performance")
    col2.dataframe(general_performance(y_true, y_score, fpr, tpr, thresholds))
    col3.subheader("Statistics at set specificity")
    specificity = col3.number_input("select specificity", value=0.97, key=key)
    col3.dataframe(performance_at_specificity(y_true, y_score, fpr, tpr, thresholds, specificity))


def self_selected_threshold(fpr: np.array, tpr: np.array, thresholds: np.array):
    cutoff_index = np.argmin(abs(tpr - (1 - fpr)))
    selected_threshold = float(st.text_input(label="input threshold", value=thresholds[cutoff_index]))
    return selected_threshold


def mega_table_filter(df: pd.DataFrame):
    st.subheader("Statistics for filtered data (based om query. you may add filtration interactivley)")
    st.text("filter the data and see statistics below")
    custom_df = AgGrid(
        df,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode="MODEL_CHANGED",
        enable_enterprise_modules=True,
        reload_data=True,
        editable=True,
    )["data"]
    y_true = custom_df["y_true"].values
    y_score = custom_df["y_score"].values
    default_threshold = 0.5
    if len(y_true) == y_true.sum():
        # all true
        tp = (y_score >= default_threshold).sum()
        fn = (y_score < default_threshold).sum()
        sensitivity = tp / (tp + fn)
        st.text(f"Only positive are in the data. sensitivity at threshold of 0.5 is {round(sensitivity, 4)}")
        st.dataframe(general_data_stats(custom_df))
    elif y_true.sum() == 0:
        # all false
        tn = (y_score <= default_threshold).sum()
        fp = (y_score > default_threshold).sum()
        specificity = tn / (tn + fp)
        st.text(f"Data has only negatives. specificity at threshold of 0.5 is {round(specificity, 4)}")
        st.dataframe(general_data_stats(custom_df))
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        label = "statistics for custom data"
        show_tables_with_statistics(custom_df, y_true, y_score, fpr, tpr, thresholds, key="filtered")
        show_plots(y_true, y_score, fpr, tpr, thresholds)


def create_sub_analysis_df(df: pd.DataFrame, selected_threshold: float, col_name: str, subgroup: str) -> pd.DataFrame:
    temp_data_stats = general_data_stats(df)
    temp_metrics = get_sn_sp_at_threshold(df["y_true"].values, df["y_score"].values, selected_threshold)
    temp_fpr, temp_tpr, temp_thresholds = metrics.roc_curve(df["y_true"].values, df["y_score"].values)
    try:
        temp_performance = general_performance(
            df["y_true"].values, df["y_score"].values, temp_fpr, temp_tpr, temp_thresholds
        )
    except ValueError:
        temp_performance = pd.DataFrame()
    concatenated = pd.concat([temp_data_stats, temp_metrics, temp_performance])
    return concatenated.rename(columns={concatenated.columns[0]: f"{col_name}:{subgroup}"})


def show_subgroup_analysis(df: pd.DataFrame, selected_threshold: float, categotical_cols: list):

    results_df = pd.DataFrame()
    histograms = {}
    for col_name in categotical_cols:
        for subgroup in df[col_name].unique():
            temp = df[df[col_name] == subgroup]
            if not temp.empty:
                temp_results = create_sub_analysis_df(temp, selected_threshold, col_name, subgroup)
                results_df = pd.concat([results_df, temp_results], axis=1)
                histograms[f"{col_name}: {subgroup}"] = plot_labeled_probabilites(
                    temp["y_true"].values, temp["y_score"].values
                )
    performance_tab, dist_tab = st.tabs(["Performance", "Distributions"])
    with performance_tab:
        AgGrid(
            results_df.T.reset_index(),
            fit_columns_on_grid_load=True,
            height=700,
            custom_css={".ag-header-cell-text": {"white-space": "pre-wrap !important"}},
        )
    with dist_tab:
        for hist_name, hist_plot in histograms.items():
            st.text(hist_name)
            hist_plot.add_vline(x=selected_threshold)
            hist_plot.update_layout(width=600, height=400)
            st.plotly_chart(hist_plot)


def show_performance(df: pd.DataFrame):
    y_true = df["y_true"].values
    y_score = df["y_score"].values
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    show_tables_with_statistics(df, y_true, y_score, fpr, tpr, thresholds)
    show_plots(y_true, y_score, fpr, tpr, thresholds)
    st.subheader("stats for self selected threshold and subgroup analysis")
    selected_threshold = self_selected_threshold(fpr, tpr, thresholds)
    st.text("insert categorical columns names (separated by ',') for sub group analysis")
    st.text("for example: sex,level")
    categotical_cols = st.text_input(label="categorical columns names (separated by ','): sex,level")
    if len(categotical_cols) > 0:
        categotical_cols = categotical_cols.split(",")
        show_subgroup_analysis(df, selected_threshold, categotical_cols)
    st.header("Customized query (see pandas query)")
    st.text("Write your own query. You can filter further using the table below.")
    st.text("If no query is passed, the whole dataset will be presented on the table")
    st.text("Query example:")
    st.text("sex=='m'")
    query = st.text_input(label="input query", key="query")
    if len(query) > 0:
        mega_table_filter(df.query(query))
    else:
        mega_table_filter(df)
