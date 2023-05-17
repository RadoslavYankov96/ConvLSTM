import numpy as np
import h5py as h5
import os
import tensorflow as tf
import re


class ImageSequenceDataset:

    def __init__(self, data_path, sequence_length, batch_size, starting_index, img_shape, fan_stack):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.starting_index = starting_index
        self.img_shape = img_shape
        self.fan_stack = fan_stack

    def load_experiment(self, file_path):
        with h5.File(file_path, 'r') as experiment:
            keys = list(experiment.keys())
            keys = sorted(keys, key=lambda s: int(re.search(r'\d+', s).group()))
            keys = keys[self.starting_index:]
            for i in range(len(keys) - 2 * self.sequence_length + 1):
                input_images = []
                target_images = []
                for k in range(self.sequence_length):
                    input_images.append(np.expand_dims(np.array(experiment[keys[i + k]], dtype=np.float32), axis=-1))
                    target_images.append(np.expand_dims(np.array(experiment[keys[i + k + self.sequence_length]],
                                                                 dtype=np.float32), axis=-1))
                fan_settings = self.get_metadata(file_path)
                input_imgs = [np.concatenate([img, fan_settings], axis=2) for img in input_images]
                '''for img in input_images:
                    img = np.concatenate([img, fan_settings], axis=2)'''
                input_sequence = np.stack(tuple(input_imgs))
                target_sequence = np.stack(tuple(target_images))
                yield input_sequence, target_sequence

    def load_data(self):
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            for input_sequences, target_sequences in self.load_experiment(file_path):
                yield input_sequences, target_sequences
    
    def get_metadata(self, file):
        name_list = list(int(i) if i.isdigit() else i for i in file.split("_"))
        fan_settings = np.array(name_list[3:6], dtype=np.float32)
        #fan_settings = np.tile(fan_settings, self.fan_stack)
        fan_settings = fan_settings/100
        fan_1 = np.full((512, 640), fan_settings[0])
        fan_2 = np.full((512, 640), fan_settings[1])
        fan_3 = np.full((512, 640), fan_settings[2])
        fan_settings = np.stack([fan_1, fan_2, fan_3], axis=2)
        #fan_settings = np.expand_dims(fan_settings, axis=0)
        return fan_settings

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.load_data, output_signature=(
            tf.TensorSpec(shape=(self.sequence_length, 512, 640, 4), dtype=tf.float32),
             #tf.TensorSpec(shape=(1, 512, 640, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(self.sequence_length, *self.img_shape), dtype=tf.float32)))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(54)

        return dataset


def main():
    dataset = ImageSequenceDataset("/home/itsnas/ueuua/BA/dataset/val/",
                                   2, 1, 4, (512, 640, 1), 30)
                                   
    dataset.create_dataset()
    '''dataset = dataset.create_dataset()
    first_five_elements = dataset.take(5)
    sequence_length = 2

    for (input_sequences, fan_settings), target_sequences in first_five_elements:
        # Input sequences
        fig, axs = plt.subplots(1, sequence_length, figsize=(10, 2))
        for i in range(sequence_length):
            axs[i].imshow(input_sequences[0, i, :, :, 0], cmap='jet')
            axs[i].axis('off')
        plt.show()'''


if __name__ == "__main__":
    main()
