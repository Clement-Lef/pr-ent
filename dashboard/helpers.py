import json
import string
from time import time

import en_core_web_lg
import inflect

# import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch

# from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Set constant values
INFLECT_ENGINE = inflect.engine()
TOP_K = 30
NLI_LIMIT = 0.9

st.set_page_config(layout="wide")


def get_top_k():
    return TOP_K


def get_nli_limit():
    return NLI_LIMIT


### Streamlit specific
@st.cache(allow_output_mutation=True)
def load_model_prompting():
    return pipeline("fill-mask", model="models/distilbert-base-uncased")


@st.cache(allow_output_mutation=True)
def load_model_nli():
    try:
        return pipeline(
            task="sentiment-analysis", model="models/roberta-large-mnli", device="mps"
        )
    except:
        return pipeline(task="sentiment-analysis", model="models/roberta-large-mnli")


@st.cache(allow_output_mutation=True)
def load_spacy_pipeline():
    return en_core_web_lg.load()


# @st.cache()
# def download_punkt():
#     nltk.download("punkt")


# download_punkt()


@st.experimental_memo(max_entries=1)
def read_json_from_web(uploaded_json):
    return json.load(uploaded_json)


@st.experimental_memo(max_entries=1)
def read_csv_from_web(uploaded_file):
    """Read CSV from the streamlit interface

    :param uploaded_file: File to read
    :type uploaded_file: UploadedFile (BytesIO)
    :return: Dataframe
    :rtype: pandas DataFrame
    """
    try:
        # Try first to read comma separated and semicolon separated files
        data = pd.read_csv(uploaded_file, sep=None, engine="python")
        # If both are not correct, then it will error and go to the except
    except pd.errors.ParserError:
        # This should be the case when there is no separator (1 column csv)
        # Reset the IO object due to the previous crash
        uploaded_file.seek(0)
        # Use standard reading of CSV (no separator)
        data = pd.read_csv(uploaded_file)
    return data


def apply_style():
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

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)


def choose_text_menu(text):
    if "text" not in st.session_state:
        st.session_state.text = "Several demonstrators were injured."
    text = st.text_area("Event description", st.session_state.text)

    return text


def initiate_widget_st_state(widget_key, perm_key, default_value):
    if perm_key not in st.session_state:
        st.session_state[perm_key] = default_value
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[perm_key]


def get_idx_column(col_name, col_list):
    if col_name in col_list:
        return col_list.index(col_name)
    else:
        return 0


def callback_add_to_multiselect(str_to_add, multiselect_key, text_input_key, *keys):
    if len(str_to_add) == 0:
        st.warning("Word is empty, did you press Enter on the field text?")
        return
    current_dict = st.session_state
    *dict_keys, item_keys = keys
    try:
        for key in dict_keys:
            current_dict = current_dict[key]
        current_dict[item_keys].append(str_to_add)
    except KeyError as e:
        raise KeyError(keys) from e

    if multiselect_key in st.session_state:
        st.session_state[multiselect_key].append(str_to_add)
    else:
        st.session_state[multiselect_key] = [str_to_add]

    st.session_state[text_input_key] = ""


# Split the text into sentences. Necessary for NLI models
def split_sentences(text):
    # return sent_tokenize(text)
    # This is cached by streamlit
    nlp_spacy = load_spacy_pipeline()
    return [str(i) for i in nlp_spacy(text).sents]


def get_num_sentences_in_list_text(list_texts):
    num_sentences = 0
    for text in list_texts:
        num_sentences += len(split_sentences(text))
    return num_sentences


###### Prompting
def query_model_prompting(model, text, prompt_with_mask, top_k, targets):
    """Query the prompting model

    :param model: Prompting model object
    :type model: Huggingface pipeline object
    :param text: Event description (context)
    :type text: str
    :param prompt_with_mask: Prompt with a mask
    :type prompt_with_mask: str
    :param top_k: Number of tokens to output
    :type top_k: integer
    :param targets: Restrict the answer to these possible tokens
    :type targets: list
    :return: Results of the prompting model
    :rtype: list of dict
    """
    sequence = text + prompt_with_mask
    with torch.no_grad():
        output_tokens = model(sequence, top_k=top_k, targets=targets)

    return output_tokens


def do_sentence_entailment(sentence, hypothesis, model):
    """Concatenate context and hypothesis then perform entailment

    :param sentence: Event description (context), 1 sentence
    :type sentence: str
    :param hypothesis: Mask filled with a token
    :type hypothesis: str
    :param model: NLI Model
    :type model: Huggingface pipeline
    :return: DataFrame containing the result of the entailment
    :rtype: pandas DataFrame
    """
    text = sentence + "</s></s>" + hypothesis
    res = model(text, return_all_scores=True)
    df_res = pd.DataFrame(res[0])
    df_res["label"] = df_res["label"].apply(lambda x: x.lower())
    df_res.columns = ["Label", "Score"]
    return df_res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_singular_form(word):
    """Get the singular form of a word

    :param word: word
    :type word: string
    :return: singular form of the word
    :rtype: string
    """
    if INFLECT_ENGINE.singular_noun(word):
        return INFLECT_ENGINE.singular_noun(word)
    else:
        return word


######### NLI + PROMPTING
def do_text_entailment(text, hypothesis, model, limit_sentence_num=2):
    """
    Do entailment for each sentence of the event description as
    model was trained on sentence pair

    :param text: Event Description (context)
    :type text: str
    :param hypothesis: Mask filled with a token
    :type hypothesis: str
    :param model: Model NLI
    :type model: Huggingface pipeline
    :return: List of entailment results for each sentence of the text
    :rtype: list
    """
    text_entailment_results = []
    for i, sentence in enumerate(split_sentences(text)[:limit_sentence_num]):
        df_score = do_sentence_entailment(sentence, hypothesis, model)
        text_entailment_results.append((sentence, hypothesis, df_score))
    return text_entailment_results


def get_true_entailment(text_entailment_results, nli_limit):
    """
    From the result of each sentence entailment, extract the maximum entailment score and
    check if it's higher than the entailment threshold.
    """
    true_hypothesis_list = []
    max_score = 0
    for sentence_entailment in text_entailment_results:
        df_score = sentence_entailment[2]
        score = df_score[df_score["Label"] == "entailment"]["Score"].values.max()
        if score > max_score:
            max_score = score
    if max_score > nli_limit:
        true_hypothesis_list.append((sentence_entailment[1], np.round(max_score, 2)))
    return list(set(true_hypothesis_list))


def run_model_nli(data, batch_size, model_nli, use_tf=False):
    if not use_tf:
        with torch.no_grad():
            return model_nli(data, top_k=3, batch_size=batch_size)
    else:
        raise NotImplementedError
        # return run_pipeline_on_gpu(data, batch_size, model_nli["tokenizer"], model_nli["model"])


def prompt_to_nli_batching(
    text,
    prompt,
    model_prompting,
    nli_model,
    nlp,
    top_k=10,
    nli_limit=0.5,
    targets=None,
    additional_words=None,
    remove_lemma=False,
    use_tf=False,
):
    # Check if text has end ponctuation
    if text[-1] not in string.punctuation:
        text += "."
    prompt_masked = prompt.format(model_prompting.tokenizer.mask_token)
    output_prompting = query_model_prompting(
        model_prompting, text, prompt_masked, top_k, targets=targets
    )
    if remove_lemma:
        output_prompting = filter_prompt_output_by_lemma(prompt, output_prompting, nlp)
    full_batch_concat = []
    prompt_tokens = []
    for token in output_prompting:
        hypothesis = prompt.format(token["token_str"])
        for i, sentence in enumerate(split_sentences(text)):
            full_batch_concat.append(sentence + "</s></s>" + hypothesis)
            prompt_tokens.append((token["token_str"], token["score"]))

    # Add words that must be tried for entailment
    # Also increase batch_size
    if additional_words:
        for i, sentence in enumerate(split_sentences(text)):
            for token in additional_words:
                hypothesis = prompt.format(token)
                full_batch_concat.append(sentence + "</s></s>" + hypothesis)
                prompt_tokens.append((token, 1))
                top_k = top_k + 1
    results_nli = run_model_nli(full_batch_concat, top_k, nli_model, use_tf)
    # Get entailed tokens
    entailed_tokens = []
    for i, res in enumerate(results_nli):
        entailed_tokens.extend(
            [
                (get_singular_form(prompt_tokens[i][0]), x["score"])
                for x in res
                if ((x["label"] == "ENTAILMENT") & (x["score"] > nli_limit))
            ]
        )
    if entailed_tokens:
        entailed_tokens = list(
            pd.DataFrame(entailed_tokens).groupby(0).max()[1].items()
        )

    return entailed_tokens, list(set(prompt_tokens))


def remove_similar_lemma_from_list(prompt, list_words, nlp):
    ## Compute a dictionnary with the lemma for all tokens
    ## If there is a duplicate lemma then the dictionnary value will be a list of the corresponding tokens
    lemma_dict = {}
    for each in list_words:
        mask_filled = nlp(prompt.strip(".").format(each))
        lemma_dict.setdefault([x.lemma_ for x in mask_filled][-1], []).append(each)

    ## Get back the list of tokens
    ## If multiple tokens available then take the shortest one
    new_token_list = []
    for key in lemma_dict.keys():
        if len(lemma_dict[key]) >= 1:
            new_token_list.append(min(lemma_dict[key], key=len))
        else:
            raise ValueError("Lemma dict has 0 corresponding words")
    return new_token_list


def filter_prompt_output_by_lemma(prompt, output_prompting, nlp):
    """
    Remove all similar lemmas from the prompt output (e.g. "protest", "protests")
    """
    list_words = [x["token_str"] for x in output_prompting]
    new_token_list = remove_similar_lemma_from_list(prompt, list_words, nlp)
    return [x for x in output_prompting if x["token_str"] in new_token_list]


# Streamlit specific run functions
@st.experimental_memo(max_entries=1024)
def do_prent(text, template, top_k, nli_limit, additional_words=None):
    """Function used to execute PRENT model

    :param text: Event text
    :type text: string
    :param template: Template with mask
    :type template: string
    :param top_k: Maximum tokens to output from prompting model
    :type top_k: int
    :param nli_limit: Threshold of entailment for NLI [0,1]
    :type nli_limit: float
    :param additional_words: List of words that bypass prompting and goes directly to NLI, defaults to None
    :type additional_words: list, optional
    :return: (Results Entailment, Results Prompting)
    :rtype: tuple
    """
    results_nli, results_pr = prompt_to_nli_batching(
        text,
        template,
        load_model_prompting(),
        load_model_nli(),
        load_spacy_pipeline(),
        top_k=top_k,
        nli_limit=nli_limit,
        targets=None,
        additional_words=additional_words,
        remove_lemma=True,
    )
    return results_nli, results_pr


def get_additional_words():
    """Extract the additional words from the codebook

    :return: list of additional words
    :rtype: list
    """
    if "add_words" in st.session_state.codebook:
        additional_words = st.session_state.codebook["add_words"]
    else:
        additional_words = None
    return additional_words


def run_prent(
    text="", templates=[], additional_words=None, progress=True, display_text=True
):
    """Execute PRENT over a list of templates and display streamlit widgets

    :param text: Event description, defaults to ""
    :type text: str, optional
    :param templates: Templates with a mask, defaults to []
    :type templates: list, optional
    :param additional_words: List of words to bypass prompting, defaults to None
    :type additional_words: list, optional
    :param progress: Display or not the progress bar, defaults to True
    :type progress: bool, optional
    :return: (results of prent, computation time)
    :rtype: tuple
    """
    # Check if there is any template and event description available
    if not templates:
        st.warning("Template list is empty. Please add one.")
        return None, None
    if not text:
        st.warning("Event description is empty.")
        return None, None

    # Display text only when computing
    if display_text:
        temp_text = st.empty()
        temp_text.markdown("**Event Descriptions:** {}".format(text))

    # Start progress bar
    if progress:
        progress_bar = st.progress(0)
    num_prent_call = len(templates)
    num_sentences = get_num_sentences_in_list_text([text])
    iter = 0
    t0 = time()

    # We set the radio choice of streamlit to Ignore at first
    if "accept_reject_text_perm" in st.session_state:
        st.session_state["accept_reject_text_perm"] = "Ignore"

    res = {}
    for template in templates:
        template = template.replace("[Z]", "{}")
        results_nli, results_pr = do_prent(
            text,
            template,
            top_k=TOP_K,
            nli_limit=NLI_LIMIT,
            additional_words=additional_words,
        )
        # Results_nli contains % of entailment, we only care about the tokens string
        res[template] = [x[0] for x in results_nli]

        # Update progress bar
        iter += 1
        if progress:
            progress_bar.progress((1 / num_prent_call) * (iter))
    if display_text:
        temp_text.markdown("")
    time_comput = (time() - t0) / num_sentences
    # This check is done otherwise the time of computation is replaced by the
    # time of computation when using cached value
    if not time_comput < st.session_state.time_comput / 5:
        st.session_state.time_comput = int(time_comput)

    # Store some results
    res["templates_used"] = templates
    res["additional_words_used"] = additional_words
    return res, time_comput


####### Find event types based on codebook and PRENT results
def check_any_conds(cond_any, list_res):
    """Function that evaluates the "OR" conditions of the codebook versus the list of filled templates

    :param cond_any: List of groundtruth filled templates
    :type cond_any: list
    :param list_res: A list of the filled templates given by PRENT
    :type list_res: list
    :return: True if any groundtruth template is inside the list given by PRENT
    :rtype: bool
    """
    cond_any = list(cond_any)
    condition = False
    # Return False if there is no any condition
    if not cond_any:
        return False
    for cond in cond_any:
        # With the current codebook design, this should never be true.
        # Before it was possible to have recursion to check AND conditions inside an OR condition
        if isinstance(cond, dict):
            condition = check_all_conds(cond["all"], list_res)
        else:
            # Check lowercase version of templates
            if cond.lower() in [x.lower() for x in list_res]:
                condition = True
                # Exit function as the other templates won't change the outcome
                return condition
    return condition


def check_all_conds(cond_all, list_res):
    """Function that evaluates the "AND" conditions of the codebook versus the list of filled templates

    :param cond_all: List of groundtruth filled templates
    :type cond_all: list
    :param list_res: A list of the filled templates given by PRENT
    :type list_res: list
    :return: True if all groundtruth template are inside the list given by PRENT
    :rtype: bool
    """
    cond_all = list(cond_all)
    # Return False if there is no all condition
    if not cond_all:
        return False
    # Start bool on True, and put it to false if any template is missing
    condition = True
    for cond in cond_all:
        # With the current codebook design, this should never be true.
        # Before it was possible to have recursion to check OR conditions inside an AND condition
        if isinstance(cond, dict):
            condition = check_any_conds(cond["any"])
        else:
            # Check lowercase version of templates
            if not (cond.lower() in [x.lower() for x in list_res]):
                condition = False
                # Exit function as the other templates won't change the outcome
                return condition
    return condition


def find_event_types(codebook, list_res):
    """This function evaluates the codebook and then outputs a list of events types corresponding to the given results of PRENT (list of filled templates).

    :param codebook: A codebook in the format given by the dashboard
    :type codebook: dict
    :param list_res: A list of the filled templates given by PRENT
    :type list_res: list
    :return: List of event type
    :rtype: list
    """
    list_event_type = []
    # Iterate over all defined event types
    for event_type in codebook["events"]:
        code_event = codebook["events"][event_type]

        is_not_all_event, is_not_any_event, is_not_event = False, False, False
        is_all_event, is_any_event, is_event = False, False, False

        # First check if NOT conditions are met
        # e.g. a filled template that is contrary to the event is present
        if "not_all" in code_event:
            cond_all = code_event["not_all"]
            if check_all_conds(cond_all, list_res):
                is_not_all_event = True
        if "not_any" in code_event:
            cond_any = code_event["not_any"]
            if check_any_conds(cond_any, list_res):
                is_not_any_event = True

        # Next we need to check if the "not_all" and "not_any" are related
        # by an "OR" or "AND".
        # This latest case needs special care because one of two list can
        # be empty so False
        if code_event["not_all_any_rel"] == "AND":
            if is_not_all_event and (not code_event["not_any"]):
                # If all TRUE and ANY is empty (so false)
                is_not_event = True
            elif is_not_any_event and (not code_event["not_all"]):
                # If any TRUE and ALL is empty (so false)
                is_not_event = True
            if is_not_all_event and is_not_any_event:
                is_not_event = True
        elif code_event["not_all_any_rel"] == "OR":
            if is_not_all_event or is_not_any_event:
                is_not_event = True

        # The other checks are not necessary if this is true, so we go
        # to the next iteration
        if is_not_event:
            continue

        # Similar to the previous checks but this time we look for templates that should be present
        if "all" in code_event:
            cond_all = code_event["all"]
            ## Then check if All conditions are met, if not exit
            if check_all_conds(cond_all, list_res):
                is_all_event = True
        if "any" in code_event:
            ## Finally check if Any conditions is met, if not exit
            cond_any = code_event["any"]
            if check_any_conds(cond_any, list_res):
                is_any_event = True

        # This case needs special care because one of two list can
        # be empty so False
        if code_event["all_any_rel"] == "AND":
            if is_all_event and (not code_event["any"]):
                # If all TRUE and ANY is empty (so false)
                is_event = True
            elif is_any_event and (not code_event["all"]):
                # If any TRUE and ALL is empty (so false)
                is_event = True
            elif is_all_event and is_any_event:
                is_event = True
        elif code_event["all_any_rel"] == "OR":
            if is_all_event or is_any_event:
                is_event = True

        # If all checks are correct, then we can add the event type to the output list
        if is_event:
            list_event_type.append(event_type)

    return list_event_type
