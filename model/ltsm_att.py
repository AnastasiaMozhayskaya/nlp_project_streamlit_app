import json
import numpy as np
from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torchutils as tu
from torchmetrics import Accuracy
from torchmetrics.functional import f1_score
from string import punctuation
import time

with open('model/word_dict.json', 'r') as fp:
    vocab_to_int = json.load(fp)
    
wv = Word2Vec.load("model/word2vec_for_ltsm.model")
VOCAB_SIZE = len(vocab_to_int)+1
HIDDEN_SIZE = 128
SEQ_LEN = 128
DEVICE='cpu'
EMBEDDING_DIM = 128 

embedding_matrix = torch.load('model/embedding_matrix.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

class ConcatAttention(nn.Module):
    def __init__(
            self, 
            hidden_size: torch.Tensor = HIDDEN_SIZE
            ) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.align  = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh   = nn.Tanh()

    def forward(
            self, 
            lstm_outputs: torch.Tensor, # BATCH_SIZE x SEQ_LEN x HIDDEN_SIZE
            final_hidden: torch.Tensor  # BATCH_SIZE x HIDDEN_SIZE
            ) -> tuple[torch.Tensor]:
        
        att_weights = self.linear(lstm_outputs)
        att_weights = torch.bmm(att_weights, final_hidden.unsqueeze(2))
        att_weights = F.softmax(att_weights.squeeze(2), dim=1)
        
        cntxt       = torch.bmm(lstm_outputs.transpose(1, 2), att_weights.unsqueeze(2))
        concatted   = torch.cat((cntxt, final_hidden.unsqueeze(2)), dim=1)
        att_hidden  = self.tanh(self.align(concatted.squeeze(-1)))

        return att_hidden, att_weights
    
class LSTMConcatAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = embedding_layer
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.attn = ConcatAttention(HIDDEN_SIZE)
        self.clf = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(att_hidden)
        return out, att_weights

model_concat = LSTMConcatAttention()
model_concat.load_state_dict(torch.load('model/lstm_att_weight.pt', map_location='cpu'))
model_concat.eval()

def pred(text): 
    start_time = time.time()
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    text = [vocab_to_int[word] for word in text.split() if vocab_to_int.get(word)]
    if len(text) <= 128:
        zeros = list(np.zeros(128 - len(text)))
        text = zeros + text
    else:
        text = text[: 128]
    text = torch.Tensor(text)
    text = text.unsqueeze(0)
    text = text.type(torch.LongTensor)
    # print(text.shape)
    pred = model_concat(text)[0].argmax(1)
    labels = {0: 'Негативный', 1:'Позитивный', 2:'Нейтральный'}
    end_time = time.time()
    inference_time = end_time - start_time
    # return labels[pred.item()], inference_time
    return f"***{labels[pred.item()]}***, время предсказания: ***{inference_time:.4f} сек***."
