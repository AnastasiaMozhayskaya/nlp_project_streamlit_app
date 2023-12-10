import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import joblib
from time import time

dict = {0: 'Нейтральный', 1: 'Положительный', 2: 'Отрицательный'}
def preprocess_bert(text):
    start_time = time()
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    embeddings = embeddings.detach().cpu().numpy()

    lr = LogisticRegression()
    lr = joblib.load('model/lr_weights.pkl')
    # with open('model/lr_weights.pkl', 'rb') as f:
    #     lr = pickle.load(f) 
    predicted_label = lr.predict(embeddings)
    predicted_label_text = dict[predicted_label[0]]
    end_time = time()

    inference_time = end_time - start_time
    return f"***{predicted_label_text}***, время предсказания: ***{inference_time:.4f} сек***."