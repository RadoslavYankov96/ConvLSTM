import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os
import tensorflow as tf


class ImageSequenceDataset:

    def __init__(self, data_path, sequence_length, batch_size, starting_index, image_shape):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.starting_index = starting_index
        self.image_shape = image_shape

    def load_experiment(self, file_path):
        with h5.File(file_path, 'r') as experiment:
            keys = list(experiment.keys())
            keys = keys[self.starting_index:]
            for i in range(len(keys)-2*self.sequence_length+1):
                input_images = []
                target_images = []
                for k in range(self.sequence_length):
                    input_images.append(np.array(experiment[keys[i+k]], dtype=np.float32))
                    target_images.append(np.array(experiment[keys[i+k+self.sequence_length]], dtype=np.float32))
                input_sequence = np.stack(tuple(input_images), axis=2)
                target_sequence = np.stack(tuple(target_images), axis=2)
                yield input_sequence, target_sequence

    def create_dataset(self):
        filenames = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_generator(
            lambda: self.load_experiment(x),
            output_signature=(
                tf.TensorSpec(shape=(self.sequence_length, *self.image_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(self.sequence_length, *self.image_shape), dtype=tf.float32)
            )
        ))
        dataset = dataset.batch(self.batch_size)
        return dataset


def main():
    experiments = ImageSequenceDataset("C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\",
                                       2, 4, 4, (520, 640, 1))

    dataset = experiments.create_dataset()
    print(dataset)


if __name__ == "__main__":
    main()
