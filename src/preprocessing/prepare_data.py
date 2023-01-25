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
    help="The embedding model used to vectorize data for lstm and triplet",
    default="models/embedding/embed_if_32.mdl/embed_if_32.mdl"
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

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))

def load_jsonl(input_path):
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def clean_tokenize_vectorize(data, embed_model_path, seq_length, vector_size):
    print("Cleaning data")
    # keep only syntactically correct mutated data
    parsable_data = []
    for d in data:
        try:
            cst.parse_module(d[0])
            cst.parse_module(d[1])
            parsable_data.append(d)
        except:
            pass

    print("Tokenizing data")
    tokenized_data = [(tokenize_python(c), tokenize_python(m)) for c, m in parsable_data]

    print("Shuffling")
    random.shuffle(tokenized_data)

    print("Loading the embedding model")
    embed_model = load_fasttext(embed_model_path)

    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size, seq_length)

    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size, seq_length)
    return vectorized_data

def prepare_condition_data(condition_data):
    print("Loading condition data")
    consistent_data = [stmt for stmt,l in zip(*condition_data) if l == 0.]
    inconsistent_data = [stmt for stmt, l in zip(*condition_data) if l == 1.]
    return consistent_data, inconsistent_data

def prepare_condition_triplet(condition_data):
    op_mutation_dict_c = {}
    ones = []

    for stmt, l in zip(*condition_data):  
        if l == 1.0:
            ones.append(stmt)
        else:
            op_mutation_dict_c[tuple(stmt)] = ones
            ones = []
    return op_mutation_dict_c


def prepare_message_data(message_data):
    print("Loading condition data")
    inconsistent_data = []
    consistent_data = []
    for msd in message_data:
        consistent_data.append(msd[0])
        if msd[1]!=[]:
            for m in msd[1]:
                inconsistent_data.append((msd[0][0], m))
    
    return consistent_data, inconsistent_data

def prepare_message_triplet(message_data):
    op_mutation_dict_c = {}
    for stmt, mml in zip(*message_data):  
        op_mutation_dict_c[tuple(stmt)] = [(stmt[0], m) for m in mml]
    return op_mutation_dict_c

def make_triplet_from_dict(t_dict, anchor = 'condition'):
    triplets = []
    if anchor == 'condition':
        for k in t_dict:
            a = k[0]
            p = k[1]
            for pair in t_dict[k]:
                n = pair[1]
                triplets.append((a, p, n))
    elif anchor == 'message':
        for k in t_dict:
            a = k[1]
            p = k[0]
            for pair in t_dict[k]:
                n = pair[0]
                triplets.append((a, p, n))
    return triplets

def prepare_pattern_data(pattern_data):
    print("Loading pattern data")
    condition_data = pattern_data[0]
    condition_data = [cd[1] for cd in condition_data]
    message_data = pattern_data[1]
    message_data = [md[1] for md in message_data if md!=[]]

    return condition_data, message_data

def prepare_pattern_triplet(pattern_data):
    condition_data = pattern_data[0]
    condition_triplets = [(cd[0][1], cd[0][0], cd[1][0]) for cd in condition_data]

    message_data = pattern_data[0]
    message_triplets = [(md[0][0], md[0][1], md[1][1]) for md in message_data]

    return message_triplets, condition_triplets

def prepare_codex_data(codex_data):
    return None

def prepare_tr_data(tr_data):
    tr = tr_data[0]
    tr_hard = tr_data[1]
    tr_inconsistent = []
    tr_hard_inconsistent = []
    for t in tr:
        tr_inconsistent.extend(t[1])
    for t in tr_hard:
        tr_hard_inconsistent.extedn(t[1])
    return tr_inconsistent, tr_hard_inconsistent

def prepare_tr_triplet(tr_data):
    tr = tr_data[0]
    tr_hard = tr_data[1]
    tr_inconsistent = []
    tr_hard_inconsistent = []
    for t in tr:
        tr_inconsistent.extend(t[1])
    for t in tr_hard:
        tr_hard_inconsistent.extedn(t[1])
    return tr_inconsistent, tr_hard_inconsistent

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
    if model == "bilstm":
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
            print(key)
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

        consistent_data = [c for c in consistent_data if c != []]
        inconsistent_data = [c for c in inconsistent_data if c != []]
        vectorized_consistent = clean_tokenize_vectorize(consistent_data, embed_model, length, vector)
        vectorized_inconsistent = clean_tokenize_vectorize(inconsistent_data, embed_model, length, vector)
        np.save(os.path.join(output, "vectorized_consistent.npy"), vectorized_consistent)
        np.save(os.path.join(output, "vectorized_inconsistent.npy"), vectorized_inconsistent)

    elif model == "codet5":
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
            print(key)
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

        consistent_data = [c for c in consistent_data if c != []]   
        inconsistent_data = [c for c in inconsistent_data if c != []]

        def construct_full_if(c, m):
            return "if " + c + " : " + m
        consistent_data = [construct_full_if(c, m) for c, m in consistent_data]
        inconsistent_data = [construct_full_if(c, m) for c, m in inconsistent_data]
        
        labeled_data = []

        for cd in consistent_data:
            labeled_data.append(
                {"source": cd, "target": "<START>CONSISTENT<END>"}
            )
        for icd in inconsistent_data:
            labeled_data.append(
                {"source": icd, "target":"<START>INCONSISTENT<END>"}
            )
        random.shuffle(labeled_data)
        dump_jsonl(labeled_data, os.path.join(output, "codet5_formatted_data.jsonl"))

    elif model == "triplet":
        pass