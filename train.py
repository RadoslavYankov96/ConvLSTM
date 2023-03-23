from models import NextSequencePredictor, NextSequencePredictor_v2, NextSequencePredictor_v3
from prepare_dataset import ImageSequenceDataset
from callbacks import tensorboard_cb, lr_scheduler, checkpoints, early_stopping, shuffle_cb
import tensorflow as tf

# Constants
TRAIN_PATH = "/home/itsnas/ueuua/BA/dataset/train/"
VAL_PATH = "/home/itsnas/ueuua/BA/dataset/val/"
TEST_PATH = "/home/itsnas/ueuua/BA/dataset/test/"
IMG_DIMS = (512, 640, 1)
CONSTANTS = (TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_DIMS)

def custom_absolute_error(y_true, y_pred):
    absolute_error = tf.abs(y_true - y_pred)
    return tf.reduce_sum(absolute_error)


def construct_datasets(constants):
    train_path, val_path, test_path, img_dims = constants
    # Hyper parameters
    batch_size = 4
    sequence_length = 2
    starting_index = 6 - sequence_length

    data_train = ImageSequenceDataset(train_path, sequence_length, batch_size, starting_index, img_dims)
    data_val = ImageSequenceDataset(val_path, sequence_length, batch_size, starting_index, img_dims)
    data_test = ImageSequenceDataset(test_path, sequence_length, batch_size, starting_index, img_dims)

    dataset_train = data_train.create_dataset()
    dataset_val = data_val.create_dataset()
    dataset_test = data_test.create_dataset()

    return dataset_train, dataset_val, dataset_test


def main():
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset_train, dataset_val, dataset_test = construct_datasets(CONSTANTS)

    model = NextSequencePredictor()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mape', optimizer=optimizer)
    model.fit(dataset_train, epochs=300, callbacks=[tensorboard_cb('tensorboard/training_14'),
    lr_scheduler(), early_stopping(), checkpoints('checkpoints/training_14/')],
    validation_data = dataset_val)
    
    model.summary()
    # model.evaluate(dataset_val, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    # model.save('saved_models/remote_3')  


if __name__ == "__main__":
    main()
