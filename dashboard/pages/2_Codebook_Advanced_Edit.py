import json

import streamlit as st

from helpers import apply_style, callback_add_to_multiselect, get_idx_column

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

st.write("# Codebook Edit")

st.write(
    """In this tab you can:
- Add or remove templates
- Add or remove additional answer candidates
- Modify the filled templates by adding new ones manually"""
)

if "templates" not in st.session_state.codebook:

    st.warning("No codebook loaded")
    st.stop()

st.write("## Codebook: Template")


with st.expander("Templates"):

    template = st.text_input(
        "Template with a mask [Z].",
        "This event involves [Z].",
        key="add_template_text_input",
    )
    st.button(
        "Add template",
        on_click=callback_add_to_multiselect,
        args=(
            template,
            "multiselect_templates",
            "add_template_text_input",
            "codebook",
            "templates",
        ),
    )

    if "multiselect_templates" not in st.session_state:
        st.session_state["multiselect_templates"] = st.session_state.codebook[
            "templates"
        ]

    st.write("Removed templates will be removed from the codebook.")
    templates = st.multiselect(
        "Templates",
        set(st.session_state.codebook["templates"]),
        st.session_state["multiselect_templates"],
        key="multiselect_templates",
    )
    st.session_state.codebook["templates"] = templates

st.write("## Codebook: Additional Answer Candidates")
st.write(
    """
    You can manually add answer candidates. Then they will be tested for entailment on every event
    description and every template even if they are not present in the prompting results.
    This is intended for case when the event that you try to describe is quite rare (e.g. shelling, missiles).

    **Caution**: Each word added will increase the computation time (about +3%).

    **Caution**: The PR-ENT model will always try to output the singular form of the word.
"""
)
with st.expander("Add answer candidates"):

    new_word = st.text_input(
        "Answer Candidate (1 word)", "", key="add_words_text_input"
    )
    st.button(
        "Add Word",
        on_click=callback_add_to_multiselect,
        args=(
            new_word,
            "multiselect_addwords",
            "add_words_text_input",
            "codebook",
            "add_words",
        ),
    )

    if "add_words" not in st.session_state.codebook:
        st.session_state.codebook["add_words"] = []

    if "multiselect_addwords" not in st.session_state:
        st.session_state["multiselect_addwords"] = st.session_state.codebook[
            "add_words"
        ]

    templates = st.multiselect(
        "Add Words",
        set(st.session_state.codebook["add_words"]),
        st.session_state["multiselect_addwords"],
        key="multiselect_addwords",
    )
    st.session_state.codebook["add_words"] = templates

# TODO: Change by giving a list of templates and allow only filling a word.
st.write("## Codebook: Additional Filled Templates")
st.write(
    """
    You can also manually add filled templates to the codebook. This is for the case when you know that a
    filled template could appear but you don't find corresponding events. This does not increase much the
    computation time. For example you could add `This event involves kidnapping.` if you have no kidnapping
    event in your dataset but you know it could happen.

    **Caution**: The PR-ENT model will always try to output the singular form of the word. (e.g. "Protests" -> "Protest")
"""
)
class_list = list(st.session_state.codebook["events"].keys())


if "filled_templates" not in st.session_state:
    st.session_state["filled_templates"] = []


with st.expander("Add Filled Template"):

    template_chosen = st.selectbox(
        "Choose a template:",
        st.session_state.codebook["templates"],
        # index=get_idx_column(template, st.session_state.codebook["templates"]),
        key="template_sct",
    )

    def add_template_with_word(template_chosen, new_word, key_text_input):
        if len(new_word) == 0:
            st.warning("Word is empty, did you press Enter on the field text?")
        else:
            st.session_state["filled_templates"].append(
                template_chosen.replace("[Z]", new_word)
            )
        st.session_state[key_text_input] = ""

    new_word = st.text_input("1 Word Mask", "", key="filled_template_text_input")
    if st.button(
        "Add Filled Template",
        on_click=add_template_with_word,
        args=(template_chosen, new_word, "filled_template_text_input"),
    ):
        st.write("Filled template added.")
    st.write("The template can then be selected for each class below.")


st.write("## Codebook: Event Types")

st.write(
    """
    Here you have access to all filled templates independently of the template. You can add/remove some of them for
    each event type.
"""
)


for event_type in st.session_state.codebook["events"].keys():
    for any_not_all in st.session_state.codebook["events"][event_type].keys():
        if (any_not_all == "all_any_rel") or (any_not_all == "not_all_any_rel"):
            pass
        else:
            st.session_state["filled_templates"].extend(
                st.session_state.codebook["events"][event_type][any_not_all]
            )

for event_type in class_list:
    st.session_state.codebook["events"].setdefault(event_type, {})
    event_type_chosen = event_type
    with st.expander(event_type):

        def declare_ms_codebook_edit(widget_key, codebook_key, widget_display):
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.codebook["events"][
                    event_type_chosen
                ].setdefault(codebook_key, [])

            tokens_all = st.multiselect(
                widget_display,
                set(st.session_state["filled_templates"]),
                st.session_state[widget_key],
                key=widget_key,
            )
            st.session_state.codebook["events"][event_type_chosen][
                codebook_key
            ] = tokens_all

        declare_ms_codebook_edit("ms_all_{}".format(event_type_chosen), "all", "ALL")

        st.session_state.codebook["events"][event_type_chosen][
            "all_any_rel"
        ] = st.selectbox(
            "Relation",
            ["AND", "OR"],
            index=get_idx_column(
                st.session_state.codebook["events"][event_type_chosen].setdefault(
                    "all_any_rel", "OR"
                ),
                ["AND", "OR"],
            ),
            key="select_relation_any_all_{}".format(event_type_chosen),
        )

        declare_ms_codebook_edit("ms_any_{}".format(event_type_chosen), "any", "ANY")
        declare_ms_codebook_edit(
            "ms_not_all_{}".format(event_type_chosen), "not_all", "NOT ALL"
        )

        st.session_state.codebook["events"][event_type_chosen][
            "not_all_any_rel"
        ] = st.selectbox(
            "Relation",
            ["AND", "OR"],
            index=get_idx_column(
                st.session_state.codebook["events"][event_type_chosen].setdefault(
                    "not_all_any_rel", "OR"
                ),
                ["AND", "OR"],
            ),
            key="select_relation_not_any_all_{}".format(event_type_chosen),
        )
        declare_ms_codebook_edit(
            "ms_not_any_{}".format(event_type_chosen), "not_any", "NOT ANY"
        )

        if st.button("Remove Class", key="remove_class_{}".format(event_type_chosen)):
            del st.session_state.codebook["events"][event_type_chosen]
st.write("## Codebook: Download")


st.download_button(
    label="Download codebook as JSON",
    data=json.dumps(st.session_state.codebook, indent=3).encode("ASCII"),
    file_name="codebook.json",
    mime="application/json",
)
