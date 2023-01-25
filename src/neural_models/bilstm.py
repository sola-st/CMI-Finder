import numpy as np
from keras.models import Sequential
from tensorflow.python.keras.models import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.regularizers import L1L2
from keras.models import Model
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.layers import concatenate
from keras.models import load_model
from keras import optimizers
import os


class BILSTM():

    def __init__(self, layers, timesteps, data_dim, dense = 0,hyper_params=None):
        self.layers = layers
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.hyper_params = hyper_params if hyper_params is not None else {}
        self.dense = dense
        self.model = None
        self.model_type = ""

    def create_bi_listm(self,):
        model = Sequential()
        lstm_layers = self.layers
        kreg = self.hyper_params.get("kreg", L1L2(0., 0.))
        rreg = self.hyper_params.get("rreg", L1L2(0., 0.))
        fnn_activation = self.hyper_params.get("fnn_activation", "relu")
        output_activation = self.hyper_params.get("output_activation", "sigmoid")
        layer1 = Bidirectional(LSTM(lstm_layers[0], return_sequences = True, kernel_regularizer=kreg, recurrent_regularizer=rreg), input_shape = (self.timesteps, self.data_dim))
        model.add(layer1)
        for i in range(1, len(lstm_layers) -1, 1):
            model.add(Bidirectional(LSTM(lstm_layers[i], return_sequences = True, kernel_regularizer=kreg, recurrent_regularizer=rreg)))
        model.add(Bidirectional(LSTM(lstm_layers[-1], kernel_regularizer=kreg, recurrent_regularizer=rreg)))
        if self.dense != 0:
            model.add(Dense(self.dense, activation = fnn_activation))
        model.add(Dense(1, activation = output_activation))
        self.model = model
        self.model_type = "BILSTM"
        

    def create_multi_input_lstm(self):
        kreg = self.hyper_params.get("kreg", L1L2(0., 0.))
        rreg = self.hyper_params.get("rreg", L1L2(0., 0.))
        fnn_activation = self.hyper_params.get("fnn_activation", "relu")
        output_activation = self.hyper_params.get("output_activation", "sigmoid")

        input1 = Input(shape=(self.timesteps, self.data_dim))
        lstm1_cond = Bidirectional(LSTM(self.layers[0], return_sequences = True, input_shape = (timesteps, data_dim)))(input1)
        lstm2_cond = Bidirectional(LSTM(self.layers[1]))(lstm1_cond)
        #output1 = Dense(1, activation='sigmoid')(lstm2_cond)
        
        input2 = Input(shape=(self.timesteps, self.data_dim))
        lstm1_raise = Bidirectional(LSTM(self.layers[0], return_sequences = True, input_shape=(self.timesteps, self.data_dim)))(input2)
        lstm2_raise = Bidirectional(LSTM(self.layers[1]))(lstm1_raise)
        #output2 = Dense(1, activation='sigmoid')(lstm2_raise)
        
        merge = concatenate([lstm2_cond, lstm2_raise])
        if self.dense != 0:
            dense = Dense(128, activation = fnn_activation)(merge)
            output = Dense(1, activation = output_activation)(dense)
        else:
            output = Dense(1, activation = output_activation)(merge)
        
        model = Model(inputs=[input1, input2], outputs=output)
        self.model = model
        self.model_type = "MULTI_INPUT_LSTM"


    def compile(self, verbose=True):
        loss = self.hyper_params.get("loss", 'binary_crossentropy')
        optimizer = self.hyper_params.get("optimizer", optimizers.Adam())
        metrics = self.hyper_params.get("metrics", ['binary_accuracy', 'AUC', 'Precision', 'Recall'])
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        if verbose:
            print(self.model.summary())


    def train(self, data, labels, verbose=1, check_point="", class_weight={0.: 0.5, 1.: 0.5}, validation_data = None):
        
        epochs = self.hyper_params.get("epochs", 20)
        batch_size = self.hyper_params.get("batch_size", 8)

        early_stop = self.hyper_params.get("early_stop", None)
        model_checkpoint = self.hyper_params.get("check_point", None)
        callbacks = self.hyper_params.get("callbacks", [])
        if early_stop is not None:
            callbacks.append(early_stop)
        if model_checkpoint:
            callbacks.append(check_point)
    
        history = self.model.fit(
            data, labels,validation_data=validation_data, epochs = epochs, batch_size = batch_size, verbose=verbose, callbacks=callbacks)
        self.history = history

    def save_model(self, folder):
        self.model.save(os.path.join(folder, "bilstm_{0}_{1}.mdl".format(self.timesteps, self.data_dim)))
    
    def evaluate(self):
        pass