import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, TimeDistributed, Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SequencePredictor:
    def __init__(self, kernel, num_layers, filters, input_shape, frames, pool):
        self.kernel = kernel
        self.num_layers = num_layers
        self.filters = filters
        self.input_shape = input_shape
        self.frames = frames
        self.pool = pool

    def build_model(self):
        model = keras.Sequential()
        for i in range(self.num_layers):
            model.add(ConvLSTM2D(filters=self.filters[i], kernel_size=self.kernel, padding='same', activation='tanh',
                                 recurrent_activation='hard_sigmoid', input_shape=self.input_shape[i],
                                 return_sequences=True))
            model.add(keras.layers.ReLU())
            model.add(BatchNormalization())
            model.add(MaxPooling3D(pool_size=self.pool, padding='same'))

        return model


first_model = SequencePredictor(3, 3, [16, 32, 64], [(2, 600, 400, 2), (2, 400, 200, 16),
                                                             (2, 200, 100, 32)],
                                2, (3, 3, 1))
first_model = first_model.build_model()
first_model.summary()

