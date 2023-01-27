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



parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    hlep="specifiy the name of the model you want to test, one of three options: bilstm, triplet, codet5",
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
    path_data = args.path_labels
    path_labels = args.path_labels
    save_name = args.export_name

    if model == "codet5":
        os.system(
            "python -m preprocessing.prepare_data --model codet5 --sources .temp_predict/data_map.json --output .temp_predict")
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        data = load_jsonl(".temp_predict/codet5_formatted_data.jsonl")
        predict_code_t5(model, tokenizer, data)

    elif model == "bilstm":
        
        data = np.load(path_data)
        labels = np.load(path_labels)

        bilstm = load_model(source)
        
        predictions = bilstm.predict(data, batch_size=1024)
        bilstm.evaluate(data, labels)

        np.save(os.path.join("datasets" ,save_name), predictions.ravel())
    elif model == "triplet":
        all_triplets_vectorized = np.load(path_data)
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