import datetime as datetime
import hashlib
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
    callback_add_to_multiselect,
    choose_text_menu,
    do_prent,
    find_event_types,
    get_additional_words,
    get_idx_column,
    get_nli_limit,
    get_num_sentences_in_list_text,
    get_top_k,
    initiate_widget_st_state,
    run_prent,
)

# Set constant values
TOP_K = get_top_k()
NLI_LIMIT = get_nli_limit()

### Styling
# Needs to be done first
apply_style()

# Avoid having ellipsis in the multi select options
styl = """
    <style>
        .stMultiSelect span{
            max-width: none;

        }
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)

# Set color of multiselect to red
st.markdown(
    """
    <style>
        span[data-baseweb="tag"] {
            background-color: red !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def validated_metric_per_event_types(validated_dataset):
    """Compute the accuracy metrics of the validated dataset
    for each event type. Compute True Positive, False Negative,
    True Negative, False Positive.

    :param validated_dataset: Dictionary containing results of PRENT validated by the user
    :type validated_dataset: dict
    :return: Dictionnary containing accuracy metric for all event types
    :rtype: dict
    """
    dict_acc = {}

    for key, val in validated_dataset.items():
        # Compute the event types based on the computed templates of PRENT
        pred_event_types = find_event_types(
            st.session_state.codebook, val["filled_templates"]
        )
        true_event_types = val["event_types"]
        # Compute only accuracy for accepted samples
        if val["decision"] == "Accept":
            # Iterate over all possible event types
            for event_type in st.session_state.codebook["events"].keys():
                dict_acc.setdefault(event_type, {})
                dict_acc[event_type].setdefault("TP", 0)
                dict_acc[event_type].setdefault("FN", 0)
                dict_acc[event_type].setdefault("FP", 0)
                dict_acc[event_type].setdefault("TN", 0)
                if (event_type in true_event_types) and (
                    event_type in pred_event_types
                ):
                    dict_acc[event_type]["TP"] += 1
                elif (event_type in true_event_types) and not (
                    event_type in pred_event_types
                ):
                    dict_acc[event_type]["FN"] += 1
                elif not (event_type in true_event_types) and (
                    event_type in pred_event_types
                ):
                    dict_acc[event_type]["FP"] += 1
                else:
                    dict_acc[event_type]["TN"] += 1

    # Normalize metrics
    if dict_acc:
        for event_type in st.session_state.codebook["events"].keys():
            dict_acc[event_type]["Accuracy"] = (
                dict_acc[event_type]["TP"] + dict_acc[event_type]["TN"]
            ) / (
                dict_acc[event_type]["TP"]
                + dict_acc[event_type]["TN"]
                + dict_acc[event_type]["FP"]
                + dict_acc[event_type]["FN"]
            )

    return dict_acc


def store_validated_data(
    text,
    decision,
    text_idx,
    templates,
    additional_words,
    list_event_type,
    prent_params=(TOP_K, NLI_LIMIT),
):
    """Function used to store the results of PRENT in a DataFrame and in the
    session state of Streamlit.

    :param text: Event description
    :type text: string
    :param decision: Decision of the user (Accept/Reject/Ignore)
    :type decision: string
    :param text_idx: Index of the event
    :type text_idx: int
    :param templates: List of template used
    :type templates: list
    :param additional_words: List of additional words used
    :type additional_words: list
    :param list_event_type: List of event type found by PRENT and Codebook
    :type list_event_type: list
    :param prent_params: Parameters of PRENT, defaults to (TOP_K, NLI_LIMIT)
    :type prent_params: tuple, optional
    """
    if "validated_data" not in st.session_state:
        st.session_state["validated_data"] = {}

    # Generate an index if the text is not coming from a csv
    if not text_idx:
        # Create a hash of 8 digits of the text to put as index
        data_idx = str(
            "manual_{}".format(
                int(
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    16,
                )
                % 10**8
            )
        )
    else:
        data_idx = str(text_idx)

    if data_idx not in st.session_state["validated_data"]:
        st.session_state["validated_data"][data_idx] = {}
    st.session_state["validated_data"][data_idx]["text"] = text
    st.session_state["validated_data"][data_idx]["templates"] = [
        template.replace("{}", "[Z]") for template in templates
    ]
    st.session_state["validated_data"][data_idx]["additional_words"] = additional_words
    st.session_state["validated_data"][data_idx]["event_types"] = list_event_type
    st.session_state["validated_data"][data_idx][
        "filled_templates"
    ] = list_filled_templates
    st.session_state["validated_data"][data_idx]["decision"] = decision
    st.session_state["validated_data"][data_idx]["prent_params"] = prent_params


### Initialize session state variables
if "codebook" not in st.session_state:
    st.session_state.codebook = {}
    st.session_state.codebook.setdefault("events", {})
    st.session_state.codebook["templates"] = []
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
if "recompute_all_templates" not in st.session_state:
    st.session_state.recompute_all_templates = False


def reset_computation_results():
    """Reset cached values in session state related to computations"""
    st.session_state.res = {}
    st.session_state.recompute_all_templates = True
    st.session_state["accept_reject_text_perm"] = "Ignore"
    st.session_state.rerun = True


def get_all_filled_templates(results):
    """Create the filled templates from PRENT results. Merging template with mask
    with the entailed tokens.

    :param results: Dictionary containing PRENT results
    :type results: dict
    :return: List of all entailed templates
    :rtype: list
    """
    filled_templates = []
    templates_used = [x.replace("[Z]", "{}") for x in results["templates_used"]]
    for template in templates_used:
        filled_template = [template.format(x) for x in results[template]]
        filled_templates.extend(filled_template)

    return filled_templates


# Split streamlit dashboard
col_intro_left, col_intro_righter = st.columns([8, 8])
with col_intro_left:
    st.markdown(
        """ # Codebook Design
    """
    )


def load_demo(
    codebook_path="codebook_demo.json",
    validated_data_path="validated_data_demo.json",
    csv_data_path="data_demo.csv",
):
    """Load demonstration files from disk

    :param codebook_path: path to codebook, defaults to "codebook_demo.json"
    :type codebook_path: str, optional
    :param validated_data_path: path to validated dataset, defaults to "validated_data_demo.json"
    :type validated_data_path: str, optional
    :param csv_data_path: path to raw data, defaults to "data_demo.csv"
    :type csv_data_path: str, optional
    """
    st.session_state.codebook = json.load(open(codebook_path))
    st.session_state.validated_data = json.load(open(validated_data_path))
    st.session_state.data = pd.read_csv(csv_data_path, delimiter=";")
    st.session_state.filtered_df = st.session_state.data
    st.session_state.text_column_design_perm = "Event Descriptions"
    st.session_state["multiselect_classes"] = list(
        st.session_state.codebook["events"].keys()
    )
    st.session_state.text_idx = 0
    st.session_state.text = (
        "On 23 August, a group attacked a village, abducting 6 people."
    )
    st.session_state.text_display = (
        "On 23 August, a group attacked a village, abducting 6 people."
    )
    st.session_state["text_options_valid_perm"] = "From CSV"
    st.session_state["text_options_valid"] = "From CSV"


def clear_all():
    """Cleare session state"""
    for each in st.session_state:
        del st.session_state[each]
    st.experimental_rerun()


# Add two buttons in the sidebar to load and clear the demo
with st.sidebar:
    if st.button("Load Demo"):
        load_demo()

    if st.button("Clear Demo"):
        clear_all()

    st.write("********")


with st.sidebar:
    # Next function used for callback when download
    def update_codebook_save_time():
        st.session_state.save_codebook_time = (
            datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        )

    if st.download_button(
        label="Download codebook as JSON",
        data=json.dumps(st.session_state.codebook, indent=3).encode("ASCII"),
        file_name="codebook.json",
        mime="application/json",
    ):
        update_codebook_save_time()
    if "save_codebook_time" in st.session_state:
        st.write("Saved on: " + st.session_state.save_codebook_time)


with st.sidebar:
    # Next function used for callback when download
    def update_validated_save_time():
        st.session_state.save_validated_time = (
            datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        )

    if st.download_button(
        label="Download labeled data",
        data=json.dumps(st.session_state["validated_data"], indent=3).encode("ASCII"),
        file_name="validated_data.json",
        mime="application/json",
    ):
        update_validated_save_time()
    if "save_validated_time" in st.session_state:
        st.write("Saved on: " + st.session_state.save_validated_time)

# Add text to sidebar
with st.sidebar:
    st.write("********")
    st.markdown(
        """
#### Manual:

1. Set the list of possible event types
2. Select the input mode of the data (Manual or CSV)
3. If the codebook is empty, write a default template
   - `This event involves [Z].` is a good starting point
4. Write/Select an event description
5. Run PR-ENT
6. Check the event type classification
   - If it is correct then select Accept and return to step 4.
   - If it is wrong then select Reject and populate the codebook with the appropriate filled templates. The classification is updated for each change, when it is correct, click Accept.
7. Return to step 4

#### Tips & Tricks:

- If you start a codebook from scratch, it may be easier to pass a manual text example for each event type to get a first codebook draft
- Current codebook accuracy based on labeled data can be found in the top right
- The approach does not aim for perfect accuracy and some failures can happen, e.g. some event descriptions can produce filled templates that are not satisfactory.
    """
    )

# Add accuracy table
with col_intro_righter:
    accuracy = st.empty()
    # We fill the table with the last acc to avoid having it disappearing each time
    if "acc_df" in st.session_state:
        accuracy.table(
            st.session_state.acc_df.loc["Accuracy":"Accuracy"].style.format("{:.2}")
        )
    performance_container = st.expander("Detailed Performances")


st.write("*********")
col_left, col_right = st.columns(2)

# Add widgets to add event type and choose text input
with col_intro_left:
    with st.expander("Event Types List"):
        st.markdown(
            """
            ## Select Event Types.
        """
        )

        if "class_list_perm" not in st.session_state:
            st.session_state["class_list_perm"] = []

        # Text field + button to add new event types to multiselect
        new_class = st.text_input(
            "Add a new event type", "", key="new_class_text_input"
        )
        st.button(
            "Add Class",
            on_click=callback_add_to_multiselect,
            args=(
                new_class,
                "multiselect_classes",
                "new_class_text_input",
                "class_list_perm",
            ),
        )
        # Multiselect to choose event types
        if "multiselect_classes" not in st.session_state:
            st.session_state["multiselect_classes"] = list(
                st.session_state.codebook["events"].keys()
            )
        class_list = st.multiselect(
            "Event Type List",
            set(
                st.session_state["class_list_perm"]
                + list(st.session_state.codebook["events"].keys())
            ),
            st.session_state["multiselect_classes"],
            key="multiselect_classes",
        )
        st.session_state["class_list_perm"] = class_list

    with st.expander("Select Text Input Mode (Manual, CSV)"):
        st.write(
            """
            Choose the text input of the event descriptions. Three choices:
            - Manual: One event description can be manually input
            - From CSV: If a CSV of event descriptions was provided
        """
        )

        def callback_radio_text_choice():
            st.session_state.text = ""
            st.session_state.text_display = ""

        initiate_widget_st_state(
            "text_options_valid", "text_options_valid_perm", "Manual"
        )
        st.session_state["text_options_valid_perm"] = st.radio(
            "Choose text input",
            ["Manual", "From CSV"],
            index=get_idx_column(
                st.session_state["text_options_valid"], ["Manual", "From CSV"]
            ),
            key="text_options_valid",
            on_change=callback_radio_text_choice,
            horizontal=True,
        )


with col_left:
    if st.session_state["text_options_valid_perm"] == "Manual":
        text = choose_text_menu("")
        # Reset all computations if text has changed
        if text != st.session_state.text:
            reset_computation_results()
        st.session_state.text_idx = None
        st.session_state.text = text
        st.session_state.text_display = text
    elif st.session_state["text_options_valid_perm"] == "From CSV":
        if st.button("Select Random Text"):
            sample = st.session_state.filtered_df.sample(n=1).iloc[0]
            text = sample[st.session_state["text_column_design_perm"]]
            idx = sample.name
            if text != st.session_state.text:
                reset_computation_results()
            st.session_state.text = text
            st.session_state.text_idx = idx
            st.session_state.text_display = st.session_state.text

    expected_time = st.session_state.time_comput * get_num_sentences_in_list_text(
        [st.session_state.text]
    )
    if st.button("Run PR-ENT / Expected time: {}sec".format(expected_time)):
        if "templates" in st.session_state.codebook:
            templates = st.session_state.codebook["templates"]
        else:
            templates = []
            st.warning("No template in codebook. Please add one.")

        additional_words = get_additional_words()
        st.session_state.res = {}
        res, time_comput = run_prent(st.session_state.text, templates, additional_words)
        st.session_state.res = res

    st.write("**Event Descriptions:** {}".format(st.session_state.text_display))
    ev_desc = st.empty()
    radio_empty = st.empty()

    if st.session_state.res:
        list_filled_templates = get_all_filled_templates(st.session_state.res)

        list_event_type = find_event_types(
            st.session_state.codebook, list_filled_templates
        )
        event_type_text = ev_desc.markdown(
            "**Current Event Types Classification**: {}".format(
                "; ".join(list_event_type)
            )
        )

        if "accept_reject_text_perm" not in st.session_state:
            st.session_state["accept_reject_text_perm"] = "Ignore"

        def callback_function(mod, key):
            st.session_state[mod] = st.session_state[key]

        radio_empty.radio(
            "Accept or Reject Coding",
            ["Ignore", "Accept", "Reject"],
            key="accept_reject_text",
            on_change=callback_function,
            args=(
                "accept_reject_text_perm",
                "accept_reject_text",
            ),
            index=get_idx_column(
                st.session_state["accept_reject_text_perm"],
                ["Ignore", "Accept", "Reject"],
            ),
            horizontal=True,
        )

        decision = st.session_state["accept_reject_text_perm"]
        text_idx = st.session_state.text_idx
        text = st.session_state.text
        store_validated_data(
            text,
            decision,
            text_idx,
            st.session_state.res["templates_used"],
            st.session_state.res["additional_words_used"],
            list_event_type,
            prent_params=(TOP_K, NLI_LIMIT),
        )


with col_right:

    if (
        st.session_state["accept_reject_text_perm"] == "Reject"
    ) or not st.session_state.codebook["templates"]:
        with st.expander("Add Templates + Explanation"):
            st.markdown(
                """
                ## Add Templates
            """
            )
            st.markdown(
                """
                For each template added. PR-ENT will be run on the selected text.
            """
            )

            if "templates" not in st.session_state.codebook:
                st.session_state.codebook["templates"] = []

            template = st.text_input(
                "Template with a mask [Z].", "This event involves [Z]."
            )

            if st.button("Add template"):
                if template not in st.session_state.codebook["templates"]:
                    ## Add template to codebook
                    st.session_state.codebook["templates"].append(template)

                    additional_words = get_additional_words()
                    prompt = template.replace("[Z]", "{}")
                    results_nli, _ = do_prent(
                        st.session_state.text,
                        prompt,
                        TOP_K,
                        NLI_LIMIT,
                        additional_words,
                    )
                    tokens_nli = [x[0] for x in results_nli]

                    # Update result table with new template
                    if not st.session_state["res"]:
                        st.session_state.res = {}
                        st.session_state.res["additional_words_used"] = additional_words
                        st.session_state.res["templates_used"] = []
                    st.session_state.res[prompt] = tokens_nli
                    st.session_state.res["templates_used"].append(template)
                    st.write("Template '{}' added.".format(template))
                else:
                    st.write("Template '{}' already added.".format(template))

        if st.session_state.codebook["templates"]:
            with st.expander("Populate Codebook Explanation"):
                st.markdown(
                    """
                ## Set the filled template to each class.
                For each class you can select one or more filled templates. When the evaluation will
                be made, these templates will be compared with the results of PR-ENT. There are 4 options:
                - ALL: If **ALL** of these filled templates are present in the results of PR-ENT then this event type is correct
                - ANY: If **ANY** of these filled templates is present in the results of PR-ENT then this event type is correct
                - NOT ALL: If **ALL** of these filled templates are present in the results of PR-ENT, then this event type is **not** correct
                    - e.g. You may want to remove all *explosions* events from a class *Killings*.
                - NOT ANY: If **ANY** of these filled templates is present in the results of PR-ENT, then this event type is **not** correct

                Moreover, **ANY/ALL** and **NOT ANY/ NOT ALL** can be made in relation by a **AND / OR** condition.
                """
                )

            st.write("***************")
            st.write("### Populate Codebook")
            if not class_list:
                st.warning("No event type in codebook.")

            tokens_list = get_all_filled_templates(st.session_state.res)

            for event_type in class_list:
                st.session_state.codebook["events"].setdefault(event_type, {})
                event_type_chosen = event_type
                with st.expander(event_type):

                    def declare_ms_event_templates(
                        widget_key, widget_display, codebook_key
                    ):
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = st.session_state.codebook[
                                "events"
                            ][event_type_chosen].setdefault(codebook_key, [])

                        tokens_all = st.multiselect(
                            widget_display,
                            set(
                                list(
                                    tokens_list
                                    + st.session_state.codebook["events"][
                                        event_type_chosen
                                    ].setdefault(codebook_key, [])
                                )
                            ),
                            st.session_state[widget_key],
                            key=widget_key,
                        )
                        st.session_state.codebook["events"][event_type_chosen][
                            codebook_key
                        ] = tokens_all

                    declare_ms_event_templates(
                        "ms_all_{}".format(event_type_chosen), "ALL", "all"
                    )

                    st.session_state.codebook["events"][event_type_chosen][
                        "all_any_rel"
                    ] = st.selectbox(
                        "Relation",
                        ["AND", "OR"],
                        index=get_idx_column(
                            st.session_state.codebook["events"][
                                event_type_chosen
                            ].setdefault("all_any_rel", "OR"),
                            ["AND", "OR"],
                        ),
                        key="select_relation_any_all_{}".format(event_type_chosen),
                    )

                    declare_ms_event_templates(
                        "ms_any_{}".format(event_type_chosen), "ANY", "any"
                    )

                    declare_ms_event_templates(
                        "ms_not_all_{}".format(event_type_chosen), "NOT ALL", "not_all"
                    )

                    st.session_state.codebook["events"][event_type_chosen][
                        "not_all_any_rel"
                    ] = st.selectbox(
                        "Relation",
                        ["AND", "OR"],
                        index=get_idx_column(
                            st.session_state.codebook["events"][
                                event_type_chosen
                            ].setdefault("not_all_any_rel", "OR"),
                            ["AND", "OR"],
                        ),
                        key="select_relation_not_any_all_{}".format(event_type_chosen),
                    )

                    declare_ms_event_templates(
                        "ms_not_any_{}".format(event_type_chosen), "NOT ANY", "not_any"
                    )

            # Workaround to avoid the expanders closing after first modification
            # I have no explanation for the bug
            if st.session_state.rerun:
                st.session_state.rerun = False
                st.experimental_rerun()


if "validated_data" in st.session_state:
    recompute = False
    performance_container.markdown(
        "If a new template is added, the previous labeled samples needs to be recomputed with it. The next button allows that, however it can take some time depending on the number of samples."
    )
    if performance_container.button(
        "Recompute Missing Templates", key="recompute_temp"
    ):
        prog_bar = performance_container.progress(0)
        for i, datapoint in enumerate(st.session_state["validated_data"].values()):
            if not set(st.session_state.codebook["templates"]).issubset(
                set(datapoint["templates"])
            ):
                # Get templates that are missing from results but present in codebook
                # These happens if templates are added a posteriori
                missing_templates = list(
                    set(st.session_state.codebook["templates"])
                    - set(set(datapoint["templates"]))
                )
                recompute = True
            # For now additional words are not recomputed
            if not set(st.session_state.codebook["add_words"]).issubset(
                set(datapoint["additional_words"])
            ):
                missing_add_words = list(
                    set(st.session_state.codebook["add_words"])
                    - set(set(datapoint["additional_words"]))
                )
                recompute = True
            else:
                missing_add_words = None

            if recompute:
                res, _ = run_prent(
                    datapoint["text"],
                    missing_templates,
                    missing_add_words,
                    progress=False,
                )
                datapoint["filled_templates"].extend(get_all_filled_templates(res))
                datapoint["templates"].extend(missing_templates)
            prog_bar.progress(
                (1 / len(st.session_state["validated_data"].values())) * (i + 1)
            )

    st.session_state.acc_df = pd.DataFrame(
        validated_metric_per_event_types(st.session_state["validated_data"])
    )
    accuracy.table(
        st.session_state.acc_df.loc["Accuracy":"Accuracy"].style.format("{:.2}")
    )
    performance_container.markdown("### Performances on labeled dataset")
    performance_container.dataframe(st.session_state.acc_df.style.format("{:.3}"))

if st.session_state.res:
    list_filled_templates = get_all_filled_templates(st.session_state.res)
    list_event_type = find_event_types(st.session_state.codebook, list_filled_templates)
    ev_desc.markdown(
        "**Current Event Types Classification**: {}".format("; ".join(list_event_type))
    )
