import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd
import os

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

# Сохранение весов модели
model_weights_filename = "model/rubert_tiny_toxicity_weights.pt"
torch.save(model.state_dict(), model_weights_filename)

# Сохранение весов токенизатора
tokenizer_weights_filename = "model/rubert_tiny_toxicity_tokenizer_weights.pt"
tokenizer.save_pretrained(tokenizer_weights_filename)

def text2toxicity(text, aggregate=False):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

"""
## Оценка степени токсичности сообщения
"""

st.image('images/toxy.png')

# Ввод предложения от пользователя
input_text = st.text_area("Введите предложение:", height=100)

# Обработка входных данных через модель
if input_text:
    # Вывод результатов
    my_dict = {
    'Не токсичный': (text2toxicity(input_text, False))[0],
    'Оскорбление': (text2toxicity(input_text, False))[1],
    'Непристойность': (text2toxicity(input_text, False))[2],
    'Угроза': (text2toxicity(input_text, False))[3],
    'Опасный': (text2toxicity(input_text, False))[4]
}
    # my_dict['index'] = 'your_index_value'
    # st.write({text2toxicity(input_text, False)[0]: 'non-toxic'})
    
    df = pd.DataFrame(my_dict, index=['вероятности'])
    st.dataframe(df)
    st.write(f'Вероятность токсичного комментария {text2toxicity(input_text, True)}')
