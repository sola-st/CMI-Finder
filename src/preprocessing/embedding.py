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


def embed_triplet(data, embed_model, vector_size, seq_length):
    triplet0_vec, _= vectorize_trim_pad([ad[0] for ad in data], embed_model, vector_size, seq_length)
    triplet1_vec, _= vectorize_trim_pad([ad[1] for ad in data], embed_model, vector_size, seq_length)
    triplet2_vec, _= vectorize_trim_pad([ad[2] for ad in data], embed_model, vector_size, seq_length)
    return np.array([tri for tri in zip(triplet0_vec, triplet1_vec, triplet2_vec)])