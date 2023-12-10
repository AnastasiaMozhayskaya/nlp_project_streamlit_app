import streamlit as st
import numpy as np
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import joblib

model_ml = LogisticRegression()
vectorizer = joblib.load("model/tf-idf.pkl")


def preprocess(text):
    # Убедитесь, что text - это список
    if isinstance(text, str):
        text = [text]
    # Преобразуйте текст
    text = vectorizer.transform(text)
    return text

model = model_ml
model = joblib.load("model/logistic_regression_weights.pkl")

def predict(text):
    start_time = time.time()
    text = preprocess(text)
    predicted_label = model.predict(text)
    dict = {'Bad': 'Отрицательный', 'Neutral': 'Нейтральный', 'Good': 'Положительный'}
    predicted_label_text = dict[predicted_label[0]]
    end_time = time.time()

    inference_time = end_time - start_time

    return f"***{predicted_label_text}***, время предсказания: ***{inference_time:.4f} сек***."
