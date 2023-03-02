from tensorflow import keras
from keras import layers
import numpy as np


class ConvLSTMBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvLSTMBlock, self).__init__()
        self.conv = layers.ConvLSTM2D(out_channels, kernel_size, padding='same', activation='tanh',
                                      recurrent_activation='hard_sigmoid', return_sequences=True)
        self.bn = layers.BatchNormalization()
        self.mp = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.mp(x)
        return x


class EncCLSTMBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(EncCLSTMBlock, self).__init__()
        self.CLSTM1 = ConvLSTMBlock(out_channels[0], kernel_size)
        self.CLSTM2 = ConvLSTMBlock(out_channels[1], kernel_size)
        self.CLSTM3 = ConvLSTMBlock(out_channels[2], kernel_size)
        self.CLSTM4 = ConvLSTMBlock(out_channels[3], kernel_size)
        self.CLSTM5 = ConvLSTMBlock(out_channels[4], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.CLSTM1(input_tensor)
        x = self.CLSTM2(x)
        x = self.CLSTM3(x)
        x = self.CLSTM4(x)
        x = self.CLSTM5(x)
        return x


class DeconvBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DeconvBlock, self).__init__()
        self.deconv = layers.ConvLSTM2D(out_channels, kernel_size, padding='same', activation='tanh',
                                        recurrent_activation='hard_sigmoid', return_sequences=True)
        self.bn = layers.BatchNormalization()
        self.ups = layers.UpSampling3D(size=(1, 2, 2))

    def call(self, input_tensor, training=False, **kwargs):
        x = self.deconv(input_tensor)
        x = self.bn(x, training=training)
        x = self.ups(x)
        return x


class DecCLSTMBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DecCLSTMBlock, self).__init__()
        self.DcLSTM1 = DeconvBlock(out_channels[0], kernel_size)
        self.DcLSTM2 = DeconvBlock(out_channels[1], kernel_size)
        self.DcLSTM3 = DeconvBlock(out_channels[2], kernel_size)
        self.DcLSTM4 = DeconvBlock(out_channels[3], kernel_size)
        self.DcLSTM5 = DeconvBlock(out_channels[4], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.DcLSTM1(input_tensor)
        x = self.DcLSTM2(x)
        x = self.DcLSTM3(x)
        x = self.DcLSTM4(x)
        x = self.DcLSTM5(x)
        return x


class NextSequencePredictor(keras.Model):
    """
    General Class for the next sequence predictor
    Args:
    """

    def __init__(self):
        super().__init__()
        self.encoder = EncCLSTMBlock([2, 2, 4, 8, 16], 3)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(256)
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(5120, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.rs = layers.Reshape((2, 16, 20, 8))
        self.decoder = DecCLSTMBlock([8, 8, 4, 4, 1], 3)
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs, training=False, **kwargs):
        input_sequence, fan_settings = inputs
        x = self.encoder(input_sequence)
        x = self.flat(x)
        x = layers.concatenate([x, fan_settings])
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.rs(x)
        x = self.decoder(x)
        if training:
            x = self.dropout(x, training=training)
        return x


def main():
    model = NextSequencePredictor()


if __name__ == "__main__":
    main()
