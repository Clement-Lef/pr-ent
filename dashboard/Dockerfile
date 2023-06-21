FROM python:3.10

RUN apt-get update
RUN apt-get install git-lfs
RUN git lfs install --skip-smudge

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

### Copy the models that you have locally (civilian model)
COPY models/ /app/models
### If you don't have the models locally (except the finetuned civilian) then git clone
WORKDIR /app/models
RUN if ! [ -d "/app/models/distilbert-base-uncased" ]; then git clone https://huggingface.co/distilbert-base-uncased && cd distilbert-base-uncased && git lfs pull --include "pytorch_model.bin"; fi
RUN if ! [ -d "/app/models/roberta-large-mnli" ]; then git clone https://huggingface.co/roberta-large-mnli && cd roberta-large-mnli && git lfs pull --include "pytorch_model.bin"; fi
# To save space remove every non-pytorch weights
RUN rm -rf */*.h5 && rm -rf */*.ot && rm -rf */*.msgpack && rm -rf */*.safetensors


WORKDIR /app
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run", "Data_Loading.py", "--browser.gatherUsageStats=false"]
