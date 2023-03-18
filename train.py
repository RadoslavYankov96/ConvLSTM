from models import NextSequencePredictor
import tensorflow as tf
import keras
from prepare_dataset import ImageSequenceDataset
from callbacks import tensorboard_cb, lr_scheduler, checkpoints, early_stopping
from matplotlib import pyplot as plt
import numpy as np

# Constants
TRAIN_PATH = "C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\train\\"
TEST_PATH = "C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\test\\"
IMG_DIMS = (512, 640, 1)
CONSTANTS = (TRAIN_PATH, TEST_PATH, IMG_DIMS)


def construct_datasets(constants):
    train_path, test_path, img_dims = constants
    # Hyper parameters
    batch_size = 1
    sequence_length = 2
    starting_index = 6 - sequence_length

    data_train = ImageSequenceDataset(train_path, sequence_length, batch_size, starting_index, img_dims)
    data_test = ImageSequenceDataset(test_path, sequence_length, batch_size, starting_index, img_dims)

    dataset_train = data_train.create_dataset()
    dataset_test = data_test.create_dataset()

    return dataset_train, dataset_test


def main():
    dataset_train, dataset_test = construct_datasets(CONSTANTS)

    model = NextSequencePredictor()
    #model = keras.models.load_model("saved_models//model_training_7//", compile=False)

    model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit(dataset_train, epochs=20, callbacks=[tensorboard_cb('tensorboard/blah'), lr_scheduler(),
                                                   checkpoints('checkpoints/blah')],
              validation_data=dataset_test)
    # model.evaluate(dataset_test, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    # model.save('saved_models/')'''
    preds = model.predict(dataset_test)
    model.summary()
    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = 5
    i = 1
    for pred in preds:
        for img in pred:
            img = 255 * np.array(img, dtype=np.float32)
            fig.add_subplot(rows, columns, i)
            i = i+1
            plt.imshow(img, cmap="jet")
    plt.show()


if __name__ == "__main__":
    main()
