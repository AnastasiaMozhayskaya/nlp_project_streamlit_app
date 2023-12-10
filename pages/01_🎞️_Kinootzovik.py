import streamlit as st
import pandas as pd
from model.bert import preprocess_bert
from model.ml import predict
# from model.rnn import pred
from model.ltsm_att import pred

"""
## Классификация киноотзывов
"""
st.image('images/kino.png')

st.sidebar.header('Панель инструментов :gear:')

text = st.text_area('Поле для ввода отзыва', height=300)

with st.sidebar:
    choice_model = st.radio('Выберите модель:', options=['ML-TFIDF', 'RuBert', 'LSTM(attention)'])
    
if choice_model == 'RuBert':
    if text:
        st.write(preprocess_bert(text))
        
if choice_model == 'ML-TFIDF':
    if text:
        st.write(predict(text))
        
if choice_model == 'LSTM(attention)':
    if text:
        st.write(pred(text))


data = pd.DataFrame({'Модель': ['ML-TFIDF-LogReg', 'RNN', 'RuBert-tiny2-LogReg'], 'F1-macro': [0.65, 0.57, 0.62]})
# Вывод таблицы
checkbox = st.sidebar.checkbox("Таблица f1-macro")
if checkbox:
    st.write("<h1 style='text-align: center; font-size: 20pt;'>Оценка качества моделей по метрике f1-macro</h1>", unsafe_allow_html=True)
    st.table(data)
