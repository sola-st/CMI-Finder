import argparse
import libcst as cst
from .preprocessing import tokenize_python, tokenize_triplets
import random
from .embedding import load_fasttext, vectorize_trim_pad, embed_triplet
import numpy as np
import json
import os
from data_collection.utils import run_merge_responses

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

parser.add_argument(
    "-n",
    default=1
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

def filter_parsable(data):
    parsable_data = []
    for d in data:
        try:
            cst.parse_module(d[0])
            cst.parse_module(d[1])
            parsable_data.append(d)
        except:
            pass
    return parsable_data
    
def clean_tokenize_vectorize(data, embed_model_path, seq_length, vector_size, n=1):
    print("Cleaning data")
    # keep only syntactically correct mutated data
    parsable_data = []
    parsable_data = run_merge_responses(data, filter_parsable, n_cpus_a=n)

    print("Tokenizing data")
    tokenized_data = [(tokenize_python(c), tokenize_python(m)) for c, m in parsable_data]

    print("Shuffling")
    random.shuffle(tokenized_data)

    print("Loading the embedding model")
    embed_model = load_fasttext(embed_model_path)

    print("Vectorizing data")
    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size, seq_length)

    return vectorized_data


def clean_tokenize_vectorize_no_shuffle(data, embed_model_path, seq_length, vector_size, n=1):
    print("Cleaning data")
    # keep only syntactically correct mutated data
    parsable_data = []
    parsable_data = run_merge_responses(data, filter_parsable, n_cpus_a=n)

    print("Tokenizing data")
    tokenized_data = [(tokenize_python(c), tokenize_python(m)) for c, m in parsable_data]

    print("Loading the embedding model")
    embed_model = load_fasttext(embed_model_path)

    print("Vectorizing data")
    vectorized_data, _ = vectorize_trim_pad(
        [c+m for c, m in tokenized_data if not len(c+m)> seq_length], 
        embed_model, vector_size, seq_length)

    return parsable_data, vectorized_data

def prepare_condition_data(condition_data):
    print("Loading condition data")
    consistent_data = [stmt for stmt,l in zip(*condition_data) if l == 0.]
    inconsistent_data = [stmt for stmt, l in zip(*condition_data) if l == 1.]
    return consistent_data, inconsistent_data

def prepare_condition_dict(condition_data):
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
    print("Loading message data")
    inconsistent_data = []
    consistent_data = []
    for msd in message_data:
        consistent_data.append(msd[0])
        if msd[1]!=[]:
            for m in msd[1]:
                inconsistent_data.append((msd[0][0], m))
    
    return consistent_data, inconsistent_data

def prepare_message_dict(message_data):
    op_mutation_dict_c = {}
    for msd in message_data:  
        op_mutation_dict_c[tuple(msd[0])] = [(msd[0][0], m) for m in msd[1]]
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

def make_condition_triplet(condition_data):
    cond_triplets = []
    cond_dict = prepare_condition_dict(condition_data)
    cond_triplets += make_triplet_from_dict(cond_dict, anchor="condition")
    cond_triplets += make_triplet_from_dict(cond_dict, anchor="message")

    return cond_triplets

def make_message_triplet(message_data):
    message_triplets = []
    message_dict = prepare_message_dict(message_data)
    message_triplets += make_triplet_from_dict(message_dict, anchor="condition")
    message_triplets += make_triplet_from_dict(message_dict, anchor="message")
    return message_triplets

def prepare_pattern_data(pattern_data):
    print("Loading pattern data")
    condition_data = pattern_data[0]
    condition_data = [cd[1] for cd in condition_data]
    message_data = pattern_data[1]
    message_data = [md[1] for md in message_data if md!=[]]

    return condition_data, message_data

def prepare_pattern_triplet(pattern_data):
    print("Loading pattern data")
    condition_data = pattern_data[0]
    condition_triplets = [(cd[0][1], cd[0][0], cd[1][0]) for cd in condition_data]
    message_data = pattern_data[0]
    message_triplets = [(md[0][0], md[0][1], md[1][1]) for md in message_data]
    return message_triplets, condition_triplets

def prepare_codex_data(codex_data):
    print("Loading codex data")
    return codex_data

def prepare_tr_data(tr_data):
    print("Loading token replacement data")
    tr = tr_data[0]
    tr_hard = tr_data[1]
    tr_inconsistent = []
    tr_hard_inconsistent = []
    for t in tr:
        tr_inconsistent.extend(t[1])
    for t in tr_hard:
        tr_hard_inconsistent.extend(t[1])
    return tr_inconsistent, tr_hard_inconsistent

def prepare_tr_triplet(tr_data):
    print("Loading token replacement data")
    tr = tr_data[0]
    tr_hard = tr_data[1]
    tr_inconsistent = []
    tr_hard_inconsistent = []
    for t in tr:
        tr_inconsistent.extend([(t[0][0], t[0][1], it[1]) if it[0] == t[0][0] else (t[0][1], t[0][0], it[0]) for it in t[1]])
    for t in tr_hard:
        tr_hard_inconsistent.extend([(t[0][0], t[0][1], it[1]) if it[0] == t[0][0] else (t[0][1], t[0][0], it[0]) for it in t[1]])
    return tr_inconsistent, tr_hard_inconsistent

def prepare_rm_data(rm_data):
    print("Loading random mutation data")
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
    length = int(args.length)
    vector = int(args.vector)
    n = int(args.n)
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
            if key in ["condition", "message", "pattern", "embed", "codex", "random", "consistent", "inconsistent"]:
                print(key)
                with open(data[key]) as dkl:
                    data_load = json.load(dkl)
            else:
                continue
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

        consistent_data = [tuple(cd) for cd in consistent_data]
        inconsistent_data = [tuple(cd) for cd in inconsistent_data]

        consistent_data = list(set(consistent_data))
        inconsistent_data = list(set(inconsistent_data))

        print("size of consistent data", len(consistent_data))
        print("size of inconsistentdata", len(inconsistent_data))

        if len(consistent_data) > 300000:
            print("Downsampling consistent data")
            random.shuffle(consistent_data)
            consistent_data = consistent_data[:300000]
        elif len(inconsistent_data) > 300000:
            print("Downsampling inconsistent data")
            random.shuffle(inconsistent_data)
            inconsistent_data = inconsistent_data[:300000]

        vectorized_consistent = clean_tokenize_vectorize(consistent_data, embed_model, length, vector, n=n)
        vectorized_inconsistent = clean_tokenize_vectorize(inconsistent_data, embed_model, length, vector, n=n)

        np.save(os.path.join(output, "bilstm_vectorized_consistent.npy"), vectorized_consistent)
        np.save(os.path.join(output, "bilstm_vectorized_inconsistent.npy"), vectorized_inconsistent)

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
            if key in ["condition", "message", "pattern", "embed", "codex", "random", "consistent", "inconsistent"]:
                print(key)
                with open(data[key]) as dkl:
                    data_load = json.load(dkl)
            else:
                continue
            if key in ["condition", "message"]:
                cs, ics = prepare_map[key](data_load)
                consistent_data += cs
                inconsistent_data += ics
            elif key in ["pattern", "embed"]:
                d1, d2 = prepare_map[key](data_load)
                inconsistent_data += d1
                inconsistent_data += d2
            elif key in ["consistent", "inconsistent", "random", "codex"]:
                inconsistent_data += prepare_map[key](data_load)

        consistent_data = [c for c in consistent_data if c != []]   
        inconsistent_data = [c for c in inconsistent_data if c != []]

        def construct_full_if(c, m):
            return "if " + c + " : " + m
        consistent_data = [construct_full_if(c, m) for c, m in consistent_data]
        inconsistent_data = [construct_full_if(c, m) for c, m in inconsistent_data]
        
        consistent_data = list(set(consistent_data))
        inconsistent_data = list(set(inconsistent_data))

        print("size of consistent data", len(consistent_data))
        print("size of inconsistentdata", len(inconsistent_data))

        if len(consistent_data) > 300000:
            print("Downsampling consistent data")
            random.shuffle(consistent_data)
            consistent_data = consistent_data[:300000]
        elif len(inconsistent_data) > 300000:
            print("Downsampling inconsistent data")
            random.shuffle(inconsistent_data)
            inconsistent_data = inconsistent_data[:300000]


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
        prepare_map = {
            "condition": make_condition_triplet,
            "message": make_message_triplet,
            "codex_triplet": prepare_codex_data,
            "embed": prepare_tr_triplet,
            "random_triplet": prepare_rm_data,
            "pattern": prepare_pattern_triplet,
            "consistent": prepare_consistent_data,
            "inconsistent": prepare_inconsistent_data
        }

        triplets = []
        for key in data:
            print(key)
            if key in ["condition", "message", "pattern", "embed", "codex_triplet", "random_triplet", "consistent", "inconsistent"]:
                with open(data[key]) as dkl:
                    data_load = json.load(dkl)
            else:
                continue
            if key in ["condition", "message", "codex_triplet"]:
                triplets += prepare_map[key](data_load)
            elif key in ["pattern", "embed", "random_triplet"]:
                d1, d2 = prepare_map[key](data_load)
                triplets += d1
                triplets += d2
            elif key in ["consistent", "inconsistent"]:
                triplets += prepare_map[key](data_load)

        triplets = [tuple(t) for t in triplets]
        triplets = list(set(triplets))

        random.shuffle(triplets)
        if len(triplets) > 600000:
            print("Downsampling triplets to 600000")
            triplets = triplets[:600000]
        all_triplets_tokenized = tokenize_triplets(triplets)
        print("Loading the embedding model")
        fft_model = load_fasttext(embed_model)

        all_triplets_vectorized = embed_triplet(all_triplets_tokenized,fft_model, vector, length)
        np.save(os.path.join(output, "triplet_data.npy"), all_triplets_vectorized)