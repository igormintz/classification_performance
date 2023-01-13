import pandas as pd
import streamlit as st

import performance

PAGES = {"Performance": performance}


def main():
    st.set_page_config(layout="wide")
    st.title("Automatic classification performance report")
    uploaded_file = st.file_uploader("upload a csv file", type="csv")
    st.text('csv file must contain "y_trues" and "y_socre" columns')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        performance.show_performance(df)
    use_example = st.checkbox("use example")
    if use_example:
        df = pd.read_csv("example.csv")
        performance.show_performance(df)


if __name__ == "__main__":
    main()
