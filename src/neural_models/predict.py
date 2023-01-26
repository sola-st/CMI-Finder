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


parser.add_argument(
    "--model",
    help = "which model: codet5, bilstm, triplet"
)

parser.add_argument(
    "--target",
    help="what's the type of data: folder, file, pairs"
)

parser.add_argument(
    "--source",
    help="path to source file or folder"
)

parser.add_argument(
    "--model_path",
    help="path to the model that is going to be used"
)


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    target = args.target
    source = args.source
    model_path = args.model_path

    if target in ["folder", "file"]:
        os.system("python -m data_collection.extract_functions --source {0} --output {1}".format(source, ".temp_predict"))
        os.system(
            "python -m data_collection.extract_data --source {0}/extracted_functions.json -n 8 --output {1}".format(".temp_predict", ".temp_predict"))
        with open(".temp_predict/data_map.json", 'w') as dmj:
            json.dump({"consistent": ".temp_predict/extracted_condition_message_pairs.json"}, dmj)
    elif target == "pairs":
        with open(".temp_predict/data_map.json", 'w') as dmj:
            json.dump({"consistent": source}, dmj)
        
    #if model == "bilstm":
    #    os.system(
    #        "python -m preprocessing.prepare_data --model bilstm --sources .temp_predict/data_map.json --output .temp_predict --length 64 --vector32")
    #elif model == "triplet":
    #    os.system(
    #        "python -m preprocessing.prepare_data --model triplet --sources .temp_predict/data_map.json --output .temp_predict --length 32 --vector32")
    if model == "codet5":
        os.system(
            "python -m preprocessing.prepare_data --model codet5 --sources .temp_predict/data_map.json --output .temp_predict")
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        data = load_jsonl(".temp_predict/codet5_formatted_data.jsonl")
        predict_code_t5(model, tokenizer, data)

    elif model == "bilstm":
        with open(".temp_predict/extracted_condition_message_pairs.json") as srcs:
            data = json.load(srcs)
        parsable_data, vectorized_data = clean_tokenize_vectorize_no_shuffle(data, "models/embedding/embed_if_32.mdl/embed_if_32.mdl",
                                            64, 32, 8)
        bilstm = load_model(model_path)
        print(vectorized_data.shape)
        predictions = bilstm.predict(vectorized_data, batch_size=1024)
        print()
        print("close to 0 ==> consistent, close to 1 ==> inconsistent")
        print()
        for d, p in zip(parsable_data, predictions):
            print("Pair:", d)
            print("Score:", p[0])
            print()
    elif model == "triplet":
        with open(".temp_predict/extracted_condition_message_pairs.json") as srcs:
            data = json.load(srcs)
        data_triplet = [(p[0], p[1], p[1]) for p in data]
        all_triplets_tokenized = tokenize_triplets(data_triplet)
        fft_model = load_fasttext("models/embedding/embed_if_32.mdl/embed_if_32.mdl")

        all_triplets_vectorized = embed_triplet(all_triplets_tokenized,fft_model, 32, 32)
        test_reshaped = all_triplets_vectorized.transpose(1, 0, 2, 3)
        embedding = load_model(model_path)
        anchor_embedding, positive_embedding, _ = (
            embedding(test_reshaped[0]),
            embedding(test_reshaped[1]),
            embedding(test_reshaped[2]),
        )
        ap_distance= tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        normalize_p = (tf.reduce_sum(tf.square(anchor_embedding), -1) + tf.reduce_sum(tf.square(positive_embedding), -1))/2
        ap_distance /= normalize_p
        ap_distance = np.array(ap_distance).ravel()
        print()
        for dp, dist in zip(data, ap_distance):
            print("Data point:", dp)
            print("Distance:", dist)
            print()
    else:
        raise ValueError("Unsupported model type")