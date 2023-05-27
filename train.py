from models import NextSequencePredictor
import tensorflow as tf
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

    # data_train = ImageSequenceDataset(train_path, sequence_length, batch_size, starting_index, img_dims)
    data_test = ImageSequenceDataset(test_path, sequence_length, batch_size, starting_index, img_dims)

    # dataset_train = data_train.create_dataset()
    dataset_test = data_test.create_dataset()

    return dataset_test


def main():
    dataset_test = construct_datasets(CONSTANTS)

    # model = NextSequencePredictor()
    model = tf.keras.models.load_model("saved_models//training_72//", compile=False)

    '''model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit(dataset_train, epochs=20, callbacks=[tensorboard_cb('tensorboard/blah'), lr_scheduler(),
                                                   checkpoints('checkpoints/blah')],
              validation_data=dataset_test)
    # model.evaluate(dataset_test, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    # model.save('saved_models/')'''

    '''dataset_test = dataset_test.take(1)
    dataset_test = list(dataset_test.as_numpy_iterator())
    img = dataset_test[0][0][0]
    fan_settings = tf.constant([100, 0, 80], dtype=np.float32)
    encoded = model.encoder(img)
    flat = model.flat(encoded)
    print(flat)'''
    preds = model.predict(dataset_test)
    model.summary()
    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = 6
    i = 1
    # preds = [preds]
    # for entry in preds:
    for pred in preds:
        for k, img in enumerate(pred):
            img = 255 * np.array(img, dtype=np.float32)
            fig.add_subplot(rows, columns, i)
            #np.save(f"images//img_{i}.npy", img)
            i = i+1
            plt.imshow(img, cmap="jet", vmin=30, vmax=170)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            '''if i < 11:
                    #pred = tf.expand_dims(pred, axis=0)
                    #fan_settings = tf.expand_dims(fan_settings, axis=0)
                new_pred = model((tf.expand_dims(pred, axis=0), tf.expand_dims(fan_settings, axis=0)))
                preds.append(new_pred)
                print("end of round")'''

    #fig.add_subplot(rows, columns, i)
    #plt.imshow(pred[1]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    #plt.suptitle(f'Prediction of next {i} frames', y=0.65)
    plt.show()


if __name__ == "__main__":
    main()
