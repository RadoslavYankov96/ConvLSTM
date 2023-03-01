import tensorflow as tf
from tensorflow import keras

train_writer = tf.summary.create_file_writer("logs/train/full/")
test_writer = tf.summary.create_file_writer("logs/test/")


def tensorboard_callback():
    tb_callback = keras.callbacks.TensorBoard(
        log_dir='tensorboard', histogram_freq=1
    )
    return tb_callback
