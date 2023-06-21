import os
import sys

import streamlit as st
from st_aggrid import AgGrid, DataReturnMode

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from helpers import apply_style, get_idx_column, read_csv_from_web, read_json_from_web

apply_style()

codebook = {}

# TODO: Anonymize for paper
st.markdown(
    """
    # Codebook Creation/Edition Tool based on the PR-ENT Approach.
    ### *Rethinking the Event Coding Pipeline with Prompt Entailment*; Clément Lefebvre, Niklas Stoehr
    ##### ARXIV LINK HERE
    ##### Contact: clement.lefebvre@datascience.ch
    ##### Version: 1.0
"""
)

st.markdown("***********")

st.markdown(
    """
## Data Loading
"""
)


st.markdown(
    """
    ### Upload a CSV of event descriptions.
"""
)
uploaded_file = st.file_uploader("Upload a csv file containing event descriptions")
if uploaded_file is not None:
    st.session_state.data = read_csv_from_web(uploaded_file)


if "data" in st.session_state:
    # Filter will be reset if the page is left and then used again
    loading_df = st.text("Loading data display...")
    st.write(
        """
        The below display of the data can be used to filter the data. Click on the *3 bars logo* when hovering over a column name and the filtering
        tool will appear. Filters are kept in memory on the whole dashboard as long as the `Reset Filters` button is not clicked.

        Current limitation: If a filter is set and the user change page. Then it can not be modified anymore and needs to be reset.
    """
    )
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = st.session_state.data
    if st.button("Reset Filters"):
        st.session_state.filtered_df = st.session_state.data

    st.session_state.filtered_df = AgGrid(
        st.session_state.filtered_df,
        height=400,
        data_return_mode=DataReturnMode.FILTERED,
        update_mode="MANUAL",
    )["data"]

    if "text_column_design_perm" not in st.session_state:
        st.session_state[
            "text_column_design_perm"
        ] = st.session_state.filtered_df.columns[0]

    def callback_function(mod, key):
        st.session_state[mod] = st.session_state[key]

    st.write("Select the column which contains the event descriptions.")
    st.selectbox(
        "Select the event description column:",
        st.session_state.filtered_df.columns,
        key="text_column_design",
        on_change=callback_function,
        args=("text_column_design_perm", "text_column_design"),
        index=get_idx_column(
            st.session_state["text_column_design_perm"],
            list(st.session_state.filtered_df.columns),
        ),
    )
    loading_df.text("")

    # Remove NaN Texts
    if st.button("Remove Empty Event Descriptions"):
        st.session_state.filtered_df = st.session_state.filtered_df.dropna(
            subset=[st.session_state["text_column_design_perm"]]
        )


st.write("********")
st.markdown("## Optional Upload")


st.markdown(
    """
    ### Upload a codebook if available. It needs to be in the format used in this dashboard.
"""
)
uploaded_codebook = st.file_uploader("Upload a codebook if available (OPTIONAL)")
if uploaded_codebook is not None:
    codebook = read_json_from_web(uploaded_codebook)
    st.session_state.codebook = codebook

st.markdown(
    """
    ### Upload a validated dataset (accept, reject, ignored) in the format of this dashboard.
"""
)

uploaded_validated_data = st.file_uploader(
    "Upload a json file containing validated data (OPTIONAL)"
)
if uploaded_validated_data is not None:
    st.session_state.validated_data = read_json_from_web(uploaded_validated_data)
