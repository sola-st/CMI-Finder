from sklearn import preprocessing
import numpy as np
from preprocessing.preprocessing import vectorize_if_block
from gensim.models import FastText


def vectorize_trim_pad(sequences, embd_model, embed_dim, seq_length):
    scalers = []
    trimed_stmts = []
    for seq in sequences:
        if len(seq) >= seq_length:
            seq_vec = vectorize_if_block(seq[:seq_length], embd_model)
        else:
            seq_vec = vectorize_if_block(seq, embd_model) + [np.zeros(embed_dim) for _ in range(len(seq), seq_length, 1)]
            
        scaler = preprocessing.MinMaxScaler()
        X_scaled = scaler.fit(seq_vec).transform(seq_vec)
        trimed_stmts.append(X_scaled)
    return np.array(trimed_stmts), scalers


def load_fasttext(path_to_fasttext):
    embed_model = FastText.load(path_to_fasttext)
    return embed_model