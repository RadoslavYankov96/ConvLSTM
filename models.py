from tensorflow import keras
from keras import layers
from keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D


class SequencePredictor(keras.Sequential):

    """
    General Class for the next sequence predictor
    Args:
    """

    def __init__(self, kernel, num_layers, filters, frames, input_format, pool):
        super().__init__()
        self.kernel = kernel
        self.num_layers = num_layers
        self.filters = filters
        self.frames = frames
        self.input_format = input_format
        self.pool = pool

    def build_encoder(self):

        """
        Building the encoder part of the model based on a ConvLSTM architecture for next sequence prediction.
        This function initializes the model
        """

        for i in range(self.num_layers):
            self.add(ConvLSTM2D(filters=self.filters[i], kernel_size=self.kernel, padding='same', activation='tanh',
                                recurrent_activation='hard_sigmoid', input_shape=self.input_format,
                                return_sequences=True))
            self.add(keras.layers.LeakyReLU())
            self.add(BatchNormalization())
            self.add(MaxPooling3D(pool_size=self.pool, padding='same'))

    def build_decoder(self):
        for i in range(self.num_layers-1, -1, -1):
            self.add(layers.UpSampling3D(size=self.pool))
            self.add(ConvLSTM2D(filters=self.filters[i], kernel_size=self.kernel, padding='same', activation='tanh',
                                recurrent_activation='hard_sigmoid', return_sequences=True))
            self.add(layers.LeakyReLU())
            self.add(BatchNormalization())
        self.add(ConvLSTM2D(filters=2, kernel_size=self.kernel, padding='same', activation='tanh',
                            recurrent_activation='hard_sigmoid', return_sequences=True))

    def build_model(self):
        self.build_encoder()
        self.build_decoder()



'''this is just an example for trying out
the idea is to create dictionaries with parameters 
'''
