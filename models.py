import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np



class ConvLSTMBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvLSTMBlock, self).__init__()
        self.conv = layers.ConvLSTM2D(out_channels, kernel_size, padding='same', activation='tanh',
                                      recurrent_activation='hard_sigmoid', return_sequences=True,
                                      kernel_regularizer=regularizers.L2(), strides=2)
        # self.do = layers.Dropout(0.1)
        self.bn = layers.BatchNormalization()
        # self.mp = layers.AveragePooling3D(pool_size=(1, 2, 2), padding='same')

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        '''if training:
            x = self.do(x, training=training)'''
        x = self.bn(x, training=training)
        # x = self.mp(x)
        return x


class EncCLSTMBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(EncCLSTMBlock, self).__init__()
        self.CLSTM1 = ConvLSTMBlock(out_channels[0], kernel_size)
        self.CLSTM2 = ConvLSTMBlock(out_channels[1], kernel_size)
        self.CLSTM3 = ConvLSTMBlock(out_channels[2], kernel_size)
        self.CLSTM4 = ConvLSTMBlock(out_channels[3], kernel_size)
        self.CLSTM5 = ConvLSTMBlock(out_channels[4], kernel_size)
        self.CLSTM6 = ConvLSTMBlock(out_channels[5], kernel_size)
        self.CLSTM7 = ConvLSTMBlock(out_channels[6], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.CLSTM1(input_tensor)
        x = self.CLSTM2(x)
        x = self.CLSTM3(x)
        x = self.CLSTM4(x)
        x = self.CLSTM5(x)
        x = self.CLSTM6(x)
        x = self.CLSTM7(x)
        return x


class DeconvBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DeconvBlock, self).__init__()
        self.deconv = layers.ConvLSTM2D(out_channels, kernel_size, padding='same', activation='tanh',
                                        recurrent_activation='hard_sigmoid', return_sequences=True,
                                        kernel_regularizer=regularizers.L2())
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
        self.DcLSTM6 = DeconvBlock(out_channels[5], kernel_size)
        self.DcLSTM7 = DeconvBlock(out_channels[6], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.DcLSTM1(input_tensor)
        x = self.DcLSTM2(x)
        x = self.DcLSTM3(x)
        x = self.DcLSTM4(x)
        x = self.DcLSTM5(x)
        x = self.DcLSTM6(x)
        x = self.DcLSTM7(x)
        
        return x


class NextSequencePredictor(tf.keras.Model):
    """
    General Class for the next sequence predictor
    Args:
    """

    def __init__(self):
        super().__init__()
        self.encoder = EncCLSTMBlock([32, 32, 64, 64, 128, 128, 128], 3)
        self.flat = layers.Flatten()
        '''self.dense1 = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn2 = layers.BatchNormalization()
        self.dense3 = layers.Dense(1024, activation='relu', kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()'''
        self.dense4 = layers.Dense(5120, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn4 = layers.BatchNormalization()
        self.rs = layers.Reshape((2, 4, 5, 128))
        self.decoder = DecCLSTMBlock([128, 128, 64, 32, 16, 8, 1], 3)
        # self.dropout1 = layers.Dropout(0.1)
        # self.dropout2 = layers.Dropout(0.2)
        # self.dropout3 = layers.Dropout(0.2)
        self.dropout4 = layers.Dropout(0.1)

    def call(self, inputs, training=False, **kwargs):
        input_sequence, fan_settings = inputs
        x = self.encoder(input_sequence)
        x = self.flat(x)
        x = layers.concatenate([x, fan_settings])
        '''x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.bn3(x, training=training)'''
        x = self.dense4(x)
        if training:
            x = self.dropout4(x, training=training)
        x = self.bn4(x, training=training)
        x = self.rs(x)
        x = self.decoder(x)
        
        return x


class ConvBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same', activation='relu',
                                  kernel_regularizer=regularizers.L2(), strides=2)
        self.td = layers.TimeDistributed(self.conv)
        # self.do = layers.Dropout(0.1)
        self.bn = layers.BatchNormalization()
        # self.mp = layers.AveragePooling3D(pool_size=(1, 2, 2), padding='same')

    def call(self, input_tensor, training=False, **kwargs):
        x = self.td(input_tensor)
        '''if training:
            x = self.do(x, training=training)'''
        x = self.bn(x, training=training)
        # x = self.mp(x)
        return x
        

class ConvEncBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvEncBlock, self).__init__()
        self.CB1 = ConvBlock(out_channels[0], kernel_size)
        self.CB2 = ConvBlock(out_channels[1], kernel_size)
        self.CB3 = ConvBlock(out_channels[2], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.CB1(input_tensor)
        x = self.CB2(x)
        x = self.CB3(x)

        return x
        
        
class EncCLSTMBlock_v2(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(EncCLSTMBlock_v2, self).__init__()
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


class DCBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DCBlock, self).__init__()
        self.conv_t = layers.Conv2DTranspose(out_channels, kernel_size, padding='same', activation='relu',
                                  kernel_regularizer=regularizers.L2(), strides=2)
        self.td = layers.TimeDistributed(self.conv_t)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        x = self.td(input_tensor)
        x = self.bn(x, training=training)
        return x
        
        
class DeconvBlock_v2(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DeconvBlock_v2, self).__init__()
        self.DC1 = DCBlock(out_channels[0], kernel_size)
        self.DC2 = DCBlock(out_channels[1], kernel_size)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.DC1(input_tensor)
        x = self.DC2(x)

        return x        
        
        
class NextSequencePredictor_v2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_enc = ConvEncBlock([2, 4, 8], 5)
        self.lstm_enc = EncCLSTMBlock_v2([8, 16, 16, 32, 32], 5)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(100, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(256, activation="relu")
        self.bn2 = layers.BatchNormalization()
        # self.dense3 = layers.Dense(1024)
        # self.bn3 = layers.BatchNormalization()
        self.dense4 = layers.Dense(1280, activation="relu")
        self.bn4 = layers.BatchNormalization()
        self.rs = layers.Reshape((2, 4, 5, 32))
        self.lstm_dec = DecCLSTMBlock([32, 16, 16, 8, 4], 5)
        self.deconv = DeconvBlock_v2([4, 1], 5)
        self.dropout1 = layers.Dropout(0.8)
        self.dropout2 = layers.Dropout(0.6)
        # self.dropout3 = layers.Dropout(0.6)
        self.dropout4 = layers.Dropout(0.5)
        
    def call(self, inputs, training=False, **kwargs):
        input_sequence, fan_settings = inputs
        x = self.conv_enc(input_sequence)
        x = self.lstm_enc(x)
        x = self.flat(x)
        x = layers.concatenate([x, fan_settings])
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        '''x = self.dense3(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.bn3(x, training=training)'''
        x = self.dense4(x)
        if training:
            x = self.dropout4(x, training=training)
        x = self.bn4(x, training=training)
        x = self.rs(x)
        x = self.lstm_dec(x)
        x = self.deconv(x)
        
        return x
        
        
class ConvEncBlock_v3(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvEncBlock_v3, self).__init__()
        self.CB1 = ConvBlock(out_channels[0], kernel_size)
        self.CB2 = ConvBlock(out_channels[1], kernel_size)
        self.CB3 = ConvBlock(out_channels[2], kernel_size)
        self.CB4 = ConvBlock(out_channels[3], kernel_size)
        self.CB5 = ConvBlock(out_channels[4], kernel_size)
        self.CB6 = ConvBlock(out_channels[5], kernel_size)
        

    def call(self, input_tensor, training=False, **kwargs):
        x = self.CB1(input_tensor)
        x = self.CB2(x)
        x = self.CB3(x)
        x = self.CB4(x)
        x = self.CB5(x)
        x = self.CB6(x)

        return x
        
        
class ConvLSTMBlock_v3(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(ConvLSTMBlock_v3, self).__init__()
        self.conv = layers.ConvLSTM2D(out_channels, kernel_size, padding='same', activation='tanh',
                                      recurrent_activation='hard_sigmoid', return_sequences=True,
                                      kernel_regularizer=regularizers.L2())
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        return x        
        
        
class CLSTMBlock_v3(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(CLSTMBlock_v3, self).__init__()
        self.CLSTM1 = ConvLSTMBlock_v3(out_channels[0], kernel_size)
        self.CLSTM2 = ConvLSTMBlock_v3(out_channels[1], kernel_size)
        self.CLSTM3 = ConvLSTMBlock_v3(out_channels[2], kernel_size)  
        
    def call(self, input_tensor, training=False, **kwargs):
        x = self.CLSTM1(input_tensor)
        x = self.CLSTM2(x)
        x = self.CLSTM3(x)
        
        return x     
        
        
class DeconvBlock_v3(layers.Layer):
    def __init__(self, out_channels, kernel_size):
        super(DeconvBlock_v3, self).__init__()
        self.DC1 = DCBlock(out_channels[0], kernel_size)
        self.DC2 = DCBlock(out_channels[1], kernel_size)
        self.DC3 = DCBlock(out_channels[2], kernel_size)
        self.DC4 = DCBlock(out_channels[3], kernel_size)
        self.DC5 = DCBlock(out_channels[4], kernel_size)
        self.DC6 = DCBlock(out_channels[5], kernel_size)


    def call(self, input_tensor, training=False, **kwargs):
        x = self.DC1(input_tensor)
        x = self.DC2(x)
        x = self.DC3(x)
        x = self.DC4(x)
        x = self.DC5(x)
        x = self.DC6(x)
        
       
        return x                          
        
        
class NextSequencePredictor_v3(tf.keras.Model):        
    def __init__(self):
        super().__init__()
        self.conv_enc = ConvEncBlock_v3([8, 16, 32, 64, 128, 256], 3)
        self.lstm_enc = CLSTMBlock_v3([256, 256, 256], 3)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(100, activation='relu', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        '''self.dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.dense3 = layers.Dense(1024, activation='relu', kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()'''
        self.dense4 = layers.Dense(5120*2, activation='relu', kernel_initializer='he_normal')
        self.bn4 = layers.BatchNormalization()
        self.rs = layers.Reshape((2, 8, 10, 64))
        self.lstm_dec = CLSTMBlock_v3([64, 64, 64], 3)
        self.deconv = DeconvBlock_v3([64, 32, 16, 8, 4, 1], 3)
        self.dropout1 = layers.Dropout(0.2)
        # self.dropout2 = layers.Dropout(0.6)
        # self.dropout3 = layers.Dropout(0.6)
        self.dropout4 = layers.Dropout(0.2)
        
    def call(self, inputs, training=False, **kwargs):
        input_sequence, fan_settings = inputs
        x = self.conv_enc(input_sequence)
        x = self.lstm_enc(x)
        x = self.flat(x)
        x = layers.concatenate([x, fan_settings])
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        '''x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.bn3(x, training=training)'''
        x = self.dense4(x)
        if training:
            x = self.dropout4(x, training=training)
        x = self.bn4(x, training=training)
        x = self.rs(x)
        x = self.lstm_dec(x)
        x = self.deconv(x)
        
        return x
        
        
class NextSequencePredictor_v4(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = EncCLSTMBlock([16, 32, 64, 128, 256, 512], 3)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(100, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn2 = layers.BatchNormalization()
        '''self.dense3 = layers.Dense(1024, activation='relu', kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()'''
        self.dense4 = layers.Dense(5120*2, activation='relu', kernel_initializer='he_normal',
        kernel_regularizer=regularizers.L2())
        self.bn4 = layers.BatchNormalization()
        self.rs = layers.Reshape((2, 8, 10, 64))
        self.deconv = DeconvBlock_v3([64, 32, 16, 8, 4, 1], 3)
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        # self.dropout3 = layers.Dropout(0.6)
        self.dropout4 = layers.Dropout(0.2)
        
    def call(self, inputs, training=False, **kwargs):
        input_sequence, fan_settings = inputs
        x = self.encoder(input_sequence)
        x = self.flat(x)
        x = layers.concatenate([x, fan_settings])
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        '''x = self.dense3(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.bn3(x, training=training)'''
        x = self.dense4(x)
        if training:
            x = self.dropout4(x, training=training)
        x = self.bn4(x, training=training)
        x = self.rs(x)
        x = self.deconv(x)
        
        return x
        

def main():
    model = NextSequencePredictor()


if __name__ == "__main__":
    main()
