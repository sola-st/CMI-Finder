import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
#from keras.models import Input
from keras.layers import LSTM
from keras.models import Model
#import tensorflow as tf
#from keras.utils.vis_utils import plot_model

from keras.layers import LeakyReLU

class TripletEmbedding():

    def __init__(self,layers, timesteps, data_dim):
        self.layers = layers
        self.timesteps = timesteps
        self.data_dim = data_dim


    def triplet_embedding_bilstm(self,):
        input_layer = Input(shape = (self.timesteps, self.data_dim))
        layer1 = LSTM(self.layers[0], return_sequences = True, input_shape = (self.timesteps, self.data_dim))(input_layer)
        layer2 = LSTM(self.layers[1], return_sequences = False, input_shape = (self.timesteps, self.data_dim))(layer1)
        #flat = Flatten()(att_l)
        output = Dense(self.layers[-1]*2)(layer2)
        model = Model(inputs=[input_layer], outputs=output, name = 'BILSTM_ENCODER')
        return model

    def triplet_embedding_fnn(self):
        input_l = Input(shape=(1024))
        dense2 = self.layers.Dense(128, activation="relu")(input_l)
        dense2 = self.layers.BatchNormalization()(dense2)
        output = self.layers.Dense(128)(dense2)

        embedding = Model(input_l, output, name="Embedding_Encoder")
        return embedding

class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        #normalize = (tf.reduce_sum(tf.square(anchor), -1) + tf.reduce_sum(tf.square(positive), -1) + tf.reduce_sum(tf.square(negative), -1))/3
        normalize_p = (tf.reduce_sum(tf.square(anchor), -1) + tf.reduce_sum(tf.square(positive), -1))/2
        normalize_n = (tf.reduce_sum(tf.square(anchor), -1) + tf.reduce_sum(tf.square(negative), -1))/2
        
        return (ap_distance/normalize_p, an_distance/normalize_n)


def triplet_embedding_bilstm(timesteps=32, data_dim=32, lstm_layers=[128, 128]):
    input_layer = Input(shape = (timesteps, data_dim))
    layer1 = LSTM(lstm_layers[0], return_sequences = True, input_shape = (timesteps, data_dim))(input_layer)
    layer2 = LSTM(lstm_layers[1], return_sequences = False, input_shape = (timesteps, data_dim))(layer1)
    #flat = Flatten()(att_l)
    output = Dense(lstm_layers[-1]*2)(layer2)
    
    model = Model(inputs=[input_layer], outputs=output, name = 'BILSTM_ENCODER')
    return model

def triplet_embedding_net(net = 'bilstm'):
    if net == 'fnn':
        return triplet_embedding_fnn()
    elif net == 'bilstm':
        return triplet_embedding_bilstm()

def loss_model():
    embedding = triplet_embedding_net()

    anchor_input = layers.Input(name="anchor", shape= (32, 32))
    positive_input = layers.Input(name="positive", shape= (32, 32))
    negative_input = layers.Input(name="negative", shape= (32, 32))

    distances = DistanceLayer(name='TripletLossLayer')(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )
    network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )
    return network, embedding


class TripletModel(Model):

    def __init__(self, network, embedding, margin=0.5, lambda1 = 1, lambda2 = 1):
        super(TripletModel, self).__init__()
        self.network = network
        self.embedding = embedding
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def call(self, inputs):
        return self.network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = self.lambda1 * ap_distance - self.lambda2 * an_distance
        #real_loss = loss_real_data(list(df.src_before), list(df.src_after), self.margin)
        real_loss = 0.
        loss = tf.maximum(loss + self.margin+real_loss, 0.0)
        return loss
    
    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


def create_triplet_model(margin=1, lambda1=1, lambda2=1):
    network, embedding = loss_model()
    network.summary()

    boundaries = [10*700, 20*700, 30*700, 40*700]
    values = [0.002, 0.001, 0.0004, 0.0003, 0.0001]
    learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    trimod = TripletModel(network, embedding, margin = margin, lambda1=lambda1, lambda2=lambda2)
    trimod.compile(optimizer=optimizers.Adam(learning_rate=learning_rate_fn))

    return trimod


def train(model, data, epochs, batch_size, val_prop):
    val = 200 * 1000
    reshaped = data.transpose(1,0, 2, 3)

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0005,
    patience=10,
    verbose=2,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    ) 

    size = len(reshaped[0])
    val = int(val_prop*size)
    model.fit([reshaped[0][:-val], reshaped[1][:-val], reshaped[2][:-val]], 
            validation_data = [reshaped[0][-val:], reshaped[1][-val:], reshaped[2][-val:]],
            epochs=epochs, batch_size = batch_size, callbacks=[early_stop])

    return model