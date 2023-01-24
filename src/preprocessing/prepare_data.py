import argparse
import libcst as cst
from .preprocessing import tokenize_python
import random
from .embedding import load_fasttext, vectorize_trim_pad
import numpy as np

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

def prepare_condition_data(condition_data, embed_model_path, seq_length, vector_size):
    print("Loading condition data")
    consistent_data = [stmt for stmt,l in zip(*condition_data) if l == 0.]
    inconsistent_data = [stmt for stmt, l in zip(*condition_data) if l == 1.]

    print("Cleaning condition data")
    # keep only syntactically correct mutated data
    parsable_inconsistent = []
    for d in inconsistent_data:
        try:
            cst.parse_module(d[0])
            cst.parse_module(d[1])
            parsable_inconsistent.append(d)
        except:
            pass

    parsable_consistent = []
    for d in consistent_data:
        try:
            cst.parse_module(d[0])
            cst.parse_module(d[1])
            parsable_consistent.append(d)
        except:
            pass

    print("Tokenizing condition data")
    tokenized_consistent_data = [(tokenize_python(c), tokenize_python(m)) for c, m in consistent_data]
    tokenized_inconsistent_data = [(tokenize_python(c), tokenize_python(m)) for c, m in inconsistent_data]

    print("Shuffling")
    random.shuffle(tokenized_consistent_data)
    random.shuffle(tokenized_inconsistent_data)

    print("Loading the embedding model")
    embed_model = load_fasttext(embed_model_path)

    vectorized_consistent_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_consistent_data if not len(c+m)> seq_length], 
        embed_model, vector_size)

    vectorized_inconsistent_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_inconsistent_data if not len(c+m)> seq_length], 
        embed_model, vector_size)

    
if __name__ == "__main__":

    args = parser.parse_args()

