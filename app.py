import pandas as pd
import streamlit as st

import performance

PAGES = {"Performance": performance}


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    uploaded_file = st.sidebar.file_uploader("upload a csv file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if selection == "Performance":
            page = PAGES[selection]
            page.show_performance(df)


if __name__ == "__main__":
    main()
