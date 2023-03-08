from models import NextSequencePredictor
from prepare_dataset import ImageSequenceDataset
from callbacks import tensorboard_cb, lr_scheduler, checkpoints, early_stopping

# Constants
TRAIN_PATH = "/home/itsnas/ueuua/BA/dataset/train/"
VAL_PATH = "/home/itsnas/ueuua/BA/dataset/val/"
IMG_DIMS = (512, 640, 1)
CONSTANTS = (TRAIN_PATH, VAL_PATH, IMG_DIMS)


def construct_datasets(constants):
    train_path, val_path, img_dims = constants
    # Hyper parameters
    batch_size = 8
    sequence_length = 2
    starting_index = 6 - sequence_length

    data_train = ImageSequenceDataset(train_path, sequence_length, batch_size, starting_index, img_dims)
    data_val = ImageSequenceDataset(val_path, sequence_length, batch_size, starting_index, img_dims)

    dataset_train = data_train.create_dataset()
    dataset_val = data_val.create_dataset()

    return dataset_train, dataset_val


def main():
    dataset_train, dataset_val = construct_datasets(CONSTANTS)

    model = NextSequencePredictor()
    model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit(dataset_train, epochs=100, callbacks=[tensorboard_cb('tensorboard/remote_1'),
    lr_scheduler(), early_stopping(), checkpoints('checkpoints/remote_1/')],
    validation_data = dataset_val)
    
    model.summary()
    # model.evaluate(dataset_val, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    model.save('saved_models/remote_1')


if __name__ == "__main__":
    main()
