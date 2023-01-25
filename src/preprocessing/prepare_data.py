import argparse
import libcst as cst
from .preprocessing import tokenize_python
import random
from .embedding import load_fasttext, vectorize_trim_pad
import numpy as np
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="The model is one of three options: codet5, lstm, triplet",
    required=True
)

parser.add_argument(
    "--sources",
    help="path to json file containing a dictionary mapping different data classes",
    required=True
)

parser.add_argument(
    "--output",
    help= "folder where to save the prepared data",
    required=True
)

parser.add_argument(
    "--embed_model",
    help="The embedding model used to vectorize data for lstm and triplet"
)

parser.add_argument(
    "--length",
    help="sequence length",
    default=64
)

parser.add_argument(
    "--vector",
    help="size of embedding vector",
    default=32
)

def clean_tokenize_vectorize(data, embed_model_path, seq_length, vector_size):
    print("Cleaning condition data")
    # keep only syntactically correct mutated data
    parsable_data = []
    for d in data:
        try:
            cst.parse_module(d[0])
            cst.parse_module(d[1])
            parsable_data.append(d)
        except:
            pass

    print("Tokenizing condition data")
    tokenized_data = [(tokenize_python(c), tokenize_python(m)) for c, m in parsable_data]

    print("Shuffling")
    random.shuffle(tokenized_data)

    print("Loading the embedding model")
    embed_model = load_fasttext(embed_model_path)

    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size)

    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size)
    return vectorized_data

def prepare_condition_data(condition_data):
    print("Loading condition data")
    consistent_data = [stmt for stmt,l in zip(*condition_data) if l == 0.]
    inconsistent_data = [stmt for stmt, l in zip(*condition_data) if l == 1.]

    return consistent_data, inconsistent_data

def prepare_message_data(message_data):
    print("Loading condition data")
    inconsistent_data = [stmt for stmt, l in zip(*message_data) if l == 1.]
    consistent_data = [stmt for stmt, l in zip(*message_data) if l == 0.]
    
    return consistent_data, inconsistent_data

def prepare_pattern_data(pattern_data):
    print("Loading pattern data")
    condition_data = pattern_data[0]
    message_data = pattern_data[1]

    return condition_data, message_data

def prepare_codex_data(codex_data):
    return None

def prepare_tr_data(tr_data):
    tr = tr_data[0]
    tr_hard = tr_data[1]
    return tr, tr_hard

def prepare_rm_data(rm_data):
    return rm_data

def prepare_consistent_data(consistent_data):
    return consistent_data

def prepare_inconsistent_data(inconsistent_data):
    return inconsistent_data

if __name__ == "__main__":

    args = parser.parse_args()
    model = args.model
    sources = args.sources
    output = args.output
    embed_model = args.embed_model
    length = args.length
    vector = args.vector

    with open(sources) as srcs:
        data = json.load(srcs)

    prepare_map = {
        "condition": prepare_condition_data,
        "message": prepare_message_data,
        "codex": prepare_codex_data,
        "embed": prepare_tr_data,
        "random": prepare_rm_data,
        "pattern": prepare_pattern_data,
        "consistent": prepare_consistent_data,
        "inconsistent": prepare_inconsistent_data
    }

    consistent_data = []
    inconsistent_data = []

    for key in data:
        with open(data[key]) as dkl:
            data_load = json.load(dkl)
        if key in ["condition", "message"]:
            cs, ics = prepare_map[key](data_load)
            consistent_data += cs
            inconsistent_data += ics
        elif key in ["pattern", "embed", "codex"]:
            d1, d2 = prepare_map[key](data_load)
            inconsistent_data += d1
            inconsistent_data += d2
        elif key in ["consistent", "inconsistent", "random"]:
            inconsistent_data += prepare_map[key](data_load)
        

    vectorized_consistent = clean_tokenize_vectorize(consistent_data, embed_model, length, vector)
    vectorized_inconsistent = clean_tokenize_vectorize(inconsistent_data, embed_model, length, vector)
    np.save(os.path.join(output, "vectorized_consistent.npy"), vectorized_consistent)
    np.save(os.path.join(output, "vectorized_inconsistent.npy"), vectorized_inconsistent)

    