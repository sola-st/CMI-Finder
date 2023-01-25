import argparse
from .bilstm import BILSTM
import json
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="which model to train, it could be one of the three following options: codet5, bilstm, triplet",
    required=True
)

parser.add_argument(
    "--class0",
    help="path to data file for consistent statements",
    required=True
)

parser.add_argument(
    "--class1",
    help="path to data file for inconsistent statements",
    required=True
)

parser.add_argument(
    "--config",
    help="path to file containing models configuration and hyperparams"
)

parser.add_argument(
    "--output",
    help="path to folder where to save the trained model",
    required=True
)

if __name__ == "__main__":

    args = parser.parse_args()
    model = args.model
    class0 = args.class0
    class1 = args.class1
    config = args.config
    output = args.output
    
    if model == "bilstm":
        if config is None:
            with open("src/neural_models/bilstm_default_config.json") as dbc:
                default_config = json.load(dbc)
        else:
            with open(config) as cfg:
                default_config = json.load(cfg)
        bilstm = BILSTM(default_config["layers"], 
                        default_config["timesteps"], 
                        default_config["data_dim"],
                        dense=default_config["dense"],
                        hyper_params=default_config["hyper_params"])
        bilstm.create_bi_listm()
        bilstm.compile()
        consistent = np.load(class0)
        inconsistent = np.load(class1)
        data_size =  min(inconsistent.shape[0], consistent.shape[0]) * 2
        train_data = np.zeros((data_size, bilstm.timesteps, bilstm.data_dim))
        labels = np.zeros((data_size), dtype = np.float32)
        for j in range(int(data_size/2)):
            train_data[2*j] = consistent[j]
            labels[2*j] = 0.
    
            train_data[2*j+1] = inconsistent[j]
            labels[2*j+1] = 1.
        print("The shape of training data is:", train_data.shape)
        indexes = [i for i in range(train_data.shape[0])] 
        random.shuffle(indexes)
        train_data = [train_data[i] for i in indexes]
        labels = [labels[i] for i in indexes]
        train_data = np.array(train_data)
        labels = np.array(labels)
        print("Launching training:")
        bilstm.train(train_data, labels)
        print("Training finished, saving the model")
        bilstm.save_model(output)

    elif model == "codet5":
        pass
    elif model == "triplet":
        pass
    else:
        raise ValueError("Unrecognized model, you should provide of these three: codet5, bilstm, triplet")

