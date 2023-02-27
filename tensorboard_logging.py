import tensorflow as tf
train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
