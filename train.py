from models import NextSequencePredictor
from prepare_dataset import ImageSequenceDataset
from tensorboard_logging import tensorboard_callback

# Constants
TRAIN_PATH = "C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\train\\"
TEST_PATH = "C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\test\\"
IMG_DIMS = (512, 640, 1)
constants = (TRAIN_PATH, TEST_PATH, IMG_DIMS)


def construct_datasets(params):
    train_path, test_path, img_dims = params
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
    dataset_train, dataset_test = construct_datasets(constants)

    model = NextSequencePredictor()
    model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit(dataset_train, epochs=15, callbacks=tensorboard_callback(), validation_data=dataset_test)
    model.summary()
    # model.evaluate(dataset_test, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    # model.save('saved_models/')


if __name__ == "__main__":
    main()
