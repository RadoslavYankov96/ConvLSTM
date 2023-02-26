import numpy as np
import h5py as h5
import os
import tensorflow as tf


class ImageSequenceDataset:

    def __init__(self, data_path, sequence_length, batch_size, starting_index, img_shape):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.starting_index = starting_index
        self.img_shape = img_shape

    def load_experiment(self, file_path):
        with h5.File(file_path, 'r') as experiment:
            keys = list(experiment.keys())
            keys = keys[self.starting_index:]
            for i in range(len(keys)-2*self.sequence_length+1):
                input_images = []
                target_images = []
                for k in range(self.sequence_length):
                    input_images.append(np.expand_dims(np.array(experiment[keys[i + k]], dtype=np.float32), axis=-1))
                    target_images.append(
                        np.expand_dims(np.array(experiment[keys[i + k + self.sequence_length]], dtype=np.float32),
                                       axis=-1))
                input_sequence = np.stack(tuple(input_images))
                target_sequence = np.stack(tuple(target_images))
                yield input_sequence, target_sequence

    def load_data(self):
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            for input_sequences, target_sequences in self.load_experiment(file_path):
                yield input_sequences, target_sequences

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.load_data, output_signature=(
                                                 tf.TensorSpec(shape=(self.sequence_length, *self.img_shape),
                                                               dtype=tf.float32),
                                                 tf.TensorSpec(shape=(self.sequence_length, *self.img_shape),
                                                               dtype=tf.float32)))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(buffer_size=5)
        return dataset
