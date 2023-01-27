import argparse
import os
import json
parser = argparse.ArgumentParser()
from preprocessing.prepare_data import clean_tokenize_vectorize_no_shuffle
from preprocessing.preprocessing import tokenize_triplets
from preprocessing.embedding import embed_triplet, load_fasttext
import numpy as np
from keras.models import load_model
from transformers.models.t5 import T5ForConditionalGeneration
from transformers import RobertaTokenizer
from .utils import load_jsonl
import tensorflow as tf
import torch


def predict_code_t5(model1, tokenizer, data):
    model1 = model1.to("cpu")
    index_c = 0

    for test_element in data:
        test_source = test_element['source']
        inputs = tokenizer(test_source, return_tensors='pt').input_ids
        generated_ids = model1.generate(inputs.to("cpu"), num_beams = 5, max_length = 300, num_return_sequences=1)
        for i, beam_output in enumerate(generated_ids):
            fix = tokenizer.decode(beam_output, skip_special_tokens=True)
            print(test_source)
            print("CodeT5 prediction:", fix)
            print("=================================================================================")
        index_c += 1

def score_test_t5(model1, tokenizer, test_stmts):
    consistent_scores = []
    for e in test_stmts:
        decoder_input_ids = torch.tensor([1,267]).unsqueeze(0)
        inputs = tokenizer(e['source'], return_tensors='pt').input_ids
        model1(inputs.to("cpu"), decoder_input_ids= decoder_input_ids.to("cpu"))[0]
        logits = model1(inputs.to("cpu"), decoder_input_ids= decoder_input_ids.to("cpu"))[0]
        tokens = torch.topk(logits,2)
        if tokens[1].ravel().cpu().detach().numpy()[0]==267:
            score = tokens[0].ravel().cpu().detach().numpy()[0]/(tokens[0].ravel().cpu().detach().numpy()[0]+tokens[0].ravel().cpu().detach().numpy()[1])
        else:
            score = 1-tokens[0].ravel().cpu().detach().numpy()[0]/(tokens[0].ravel().cpu().detach().numpy()[0]+tokens[0].ravel().cpu().detach().numpy()[1])
        consistent_scores.append(score)
    return consistent_scores

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="specifiy the name of the model you want to test, one of three options: bilstm, triplet, codet5",
    required=True
)

parser.add_argument(
    "--source",
    help="specify the path of the model",
    required=True
)

parser.add_argument(
    "--path_data",
    help="path to perpared data"
)

parser.add_argument(
    "--path_labels",
    help="path to labels"
)

parser.add_argument(
    "--export_name",
    help="name of predictions save file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = args.model
    source = args.source
    path_data = args.path_data
    path_labels = args.path_labels
    save_name = args.export_name

    if model == "codet5":
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        model = T5ForConditionalGeneration.from_pretrained(source)
        data = load_jsonl(path_data)
        scores = score_test_t5(model, tokenizer, data)
        np.save(os.path.join("datasets" ,save_name), np.array(scores))

    elif model == "bilstm":
        
        data = np.load(path_data)
        labels = np.load(path_labels)

        print(path_data)
        bilstm = load_model(source)
        
        print(data.shape)
        print(labels.shape)
        predictions = bilstm.predict(data, batch_size=1024)
        bilstm.evaluate(data, labels)

        np.save(os.path.join("datasets" ,save_name), predictions.ravel())

    elif model == "triplet":
        with open(path_data) as srcs:
            data = json.load(srcs)
        data_triplet = [(p[0], p[1], p[1]) for p in data[:1000]]
        all_triplets_tokenized = tokenize_triplets(data_triplet)
        fft_model = load_fasttext("models/embedding/embed_if_32.mdl/embed_if_32.mdl")

        all_triplets_vectorized = embed_triplet(all_triplets_tokenized,fft_model, 32, 32)
        test_reshaped = all_triplets_vectorized.transpose(1, 0, 2, 3)
        
        embedding = load_model(source)
        anchor_embedding, positive_embedding, _ = (
            embedding(test_reshaped[0]),
            embedding(test_reshaped[1]),
            embedding(test_reshaped[2]),
        )
        ap_distance= tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        normalize_p = (tf.reduce_sum(tf.square(anchor_embedding), -1) + tf.reduce_sum(tf.square(positive_embedding), -1))/2
        ap_distance /= normalize_p
        ap_distance = np.array(ap_distance).ravel()
        np.save(os.path.join("datasets" ,save_name), ap_distance)
    else:
        raise ValueError("Unsupported model type")