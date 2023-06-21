# Installation

## Install the library

`pip install -r requirements.txt`

## Download the models locally

To clone the models, you need git lfs (`git lfs install --skip-smudge`). The models have to be cloned in the `/models` folder. The model used are the following:

`distilbert-base-uncased`: https://huggingface.co/distilbert-base-uncased
  - `if ! [ -d "distilbert-base-uncased" ]; then git clone https://huggingface.co/distilbert-base-uncased && cd distilbert-base-uncased && git lfs pull --include "pytorch_model.bin" && cd ..; fi`

`roberta-large-mnli`: https://huggingface.co/roberta-large-mnli
  - `if ! [ -d "roberta-large-mnli" ]; then git clone https://huggingface.co/roberta-large-mnli && cd roberta-large-mnli && git lfs pull --include "pytorch_model.bin" && cd ..; fi`

## Run the dashboard

`streamlit run Data_Loading.py --browser.gatherUsageStats=false`
