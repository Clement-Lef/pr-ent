from transformers import pipeline

tmp = pipeline("fill-mask", model="distilbert-base-uncased")
tmp = pipeline("sentiment-analysis", model="roberta-large-mnli")
