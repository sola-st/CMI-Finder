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
from keras.layers.merge import concatenate

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
