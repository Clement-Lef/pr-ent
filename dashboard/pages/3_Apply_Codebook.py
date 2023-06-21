import json
import os
import sys

import pandas as pd
import streamlit as st

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from helpers import (
    apply_style,
    find_event_types,
    get_additional_words,
    get_nli_limit,
    get_num_sentences_in_list_text,
    get_top_k,
    run_prent,
)

### Styling
apply_style()


TOP_K = get_top_k()
NLI_LIMIT = get_nli_limit()


### Initialize session state variables
if "codebook" not in st.session_state:
    st.session_state.codebook = {}
    st.session_state.codebook.setdefault("events", {})

if "text" not in st.session_state:
    st.session_state.text = ""

if "res" not in st.session_state:
    st.session_state.res = None

if "accept_reject_text_perm" not in st.session_state:
    st.session_state.accept_reject_text_perm = None

if "validated_data" not in st.session_state:
    st.session_state["validated_data"] = {}

if "time_comput" not in st.session_state:
    st.session_state.time_comput = 20

if "rerun" not in st.session_state:
    st.session_state.rerun = False

if "label_res" not in st.session_state:
    st.session_state.label_res = {}

if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = pd.DataFrame()

if len(st.session_state["filtered_df"]) == 0:
    st.warning("No data loaded.")


def reset_computation_results():
    st.session_state.res = {}
    st.session_state.recompute_all_templates = True
    st.session_state["accept_reject_text_perm"] = "Ignore"
    st.session_state.rerun = True


with st.sidebar:
    st.markdown(
        "Clicking any of these button during labeling will pause the process and download the latest version."
    )
    dl_labeled_button = st.empty()
    dl_labeled_button.download_button(
        label="Download Labeled Data",
        data=st.session_state["filtered_df"].to_csv(sep=";").encode("utf-8"),
        file_name="labeled_data.csv",
        mime="text/csv",
    )

    dl_prent_button = st.empty()
    dl_prent_button.download_button(
        label="Download PR-ENT results",
        data=json.dumps(st.session_state["label_res"], indent=3).encode("ASCII"),
        file_name="prent_results.json",
        mime="application/json",
    )


st.markdown(
    """# Apply codebook to the dataset [WARNING: For large dataset (>1000 events), use the pipeline scripts]
The currently loaded codebook will be used to find the event types of all event description in the currently loaded dataset. This can take some time (minutes to hours) depending on the size of the dataset (number of events, length of text).


"""
)

markdown_num_events = st.empty()

label_button = st.empty()
st.markdown("#### Main progress bar")
main_progress_bar = st.empty()
main_progress_bar = main_progress_bar.progress(0)

st.markdown("#### Last labeled event")
temp_text = st.empty()
temp_class = st.empty()
temp_text.markdown("**Event Descriptions:** {}".format(""))
temp_class.markdown("**Event Types Classification**: {}".format(""))
st.markdown(
    """#### Pause/Stop the event coding
Pressing the button once will stop the process at the next iteration."""
)
stop_button = st.button("Stop")

for event_type in st.session_state.codebook["events"]:
    if event_type not in st.session_state.filtered_df.columns:
        st.session_state.filtered_df[event_type] = 0

expected_time = 0
num_sentences = 0
for idx in st.session_state.filtered_df.index:
    subsampled_data = st.session_state.filtered_df.loc[idx:idx]
    list_text = subsampled_data[st.session_state["text_column_design_perm"]].values[:1]
    list_index = subsampled_data.index[:1]
    if list_text[0] != st.session_state.text:
        reset_computation_results()
    st.session_state.text = list_text[0]
    num_sentences += get_num_sentences_in_list_text([st.session_state.text])
    expected_time += st.session_state.time_comput * get_num_sentences_in_list_text(
        [st.session_state.text]
    )

markdown_num_events.markdown(
    "Number of events: {} ¦ Number of sentences: {}".format(
        len(st.session_state.filtered_df.index), num_sentences
    )
)


if label_button.button(
    "Label Data", disabled=len(st.session_state["filtered_df"]) == 0
):
    num_text = 0
    main_progress_bar.progress(num_text)
    temp_text.markdown("")
    temp_class.markdown("")
    tot_num_text = len(st.session_state.filtered_df.index)

    for idx in st.session_state.filtered_df.index:
        subsampled_data = st.session_state.filtered_df.loc[idx:idx]
        list_text = subsampled_data[st.session_state["text_column_design_perm"]].values[
            :1
        ]
        list_index = subsampled_data.index[:1]
        if list_text[0] != st.session_state.text:
            reset_computation_results()
        st.session_state.text = list_text[0]
        st.session_state.text_idx = list_index[0]
        st.session_state.template_list = []
        st.session_state.text_display = st.session_state.text

        st.session_state.res = {}
        res, time_comput = run_prent(
            st.session_state.text,
            st.session_state.codebook["templates"],
            get_additional_words(),
            progress=False,
            display_text=False,
        )
        st.session_state.res = res

        list_filled_templates = []
        for template in st.session_state.res:
            tmp = template.replace("[Z]", "{}")
            list_filled_templates.extend(
                [tmp.format(x) for x in st.session_state.res[template]]
            )
        list_event_type = find_event_types(
            st.session_state.codebook, list_filled_templates
        )
        for event_type in list_event_type:
            st.session_state.filtered_df.loc[idx, event_type] = 1
        temp_text.markdown(
            "**Event Descriptions:** {}".format(st.session_state.text_display)
        )
        temp_class.markdown(
            "**Event Types Classification**: {}".format("; ".join(list_event_type))
        )

        # Save results
        st.session_state.label_res[st.session_state.text_display] = {}
        st.session_state.label_res[st.session_state.text_display][
            "prent_results"
        ] = st.session_state.res
        st.session_state.label_res[st.session_state.text_display]["prent_params"] = (
            TOP_K,
            NLI_LIMIT,
        )
        st.session_state.label_res[st.session_state.text_display][
            "event_types"
        ] = list_event_type

        num_text += 1
        main_progress_bar.progress(num_text / tot_num_text)

    # Need to update the buttons otherwise it doesn't update the downloaded file
    # and the user would need to click two times
    dl_labeled_button.download_button(
        label="Download Labeled Data",
        data=st.session_state["filtered_df"].to_csv(sep=";").encode("utf-8"),
        file_name="labeled_data.csv",
        mime="text/csv",
        key="tmp",
    )

    dl_prent_button.download_button(
        label="Download PR-ENT results",
        data=json.dumps(st.session_state["label_res"], indent=3).encode("ASCII"),
        file_name="prent_results.json",
        mime="application/json",
    )
