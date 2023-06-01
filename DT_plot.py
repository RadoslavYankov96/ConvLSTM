from models import NextSequencePredictor, NextSequencePredictor_v4, NextSequencePredictor_v3
from prepare_dataset import ImageSequenceDataset
from callbacks import tensorboard_cb, lr_scheduler, checkpoints, early_stopping, shuffle_cb
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


# Constants
TRAIN_PATH = "/home/itsnas/ueuua/BA/dataset/train/"
VISUALIZE_PATH = "/home/itsnas/ueuua/BA/dataset/visualize/"
TEST_PATH = "/home/itsnas/ueuua/BA/dataset/test/"
IMG_DIMS = (512, 640, 1)
CONSTANTS = (TRAIN_PATH, VISUALIZE_PATH, TEST_PATH, IMG_DIMS)

def custom_absolute_error(y_true, y_pred):
    absolute_error = tf.abs(y_true - y_pred)
    return tf.reduce_sum(absolute_error)


def construct_datasets(constants):
    train_path, visualize_path, test_path, img_dims = constants
    # Hyper parameters
    batch_size = 16
    sequence_length = 2
    starting_index = 6 - sequence_length
    fan_stack = 100
    
    data_visualize = ImageSequenceDataset(visualize_path, sequence_length, batch_size, starting_index, img_dims, fan_stack)

    '''data_train = ImageSequenceDataset(train_path, sequence_length, batch_size, starting_index, img_dims, fan_stack)
    data_val = ImageSequenceDataset(val_path, sequence_length, batch_size, starting_index, img_dims, fan_stack)
    data_test = ImageSequenceDataset(test_path, sequence_length, batch_size, starting_index, img_dims, fan_stack)
    
    dataset_train = data_train.create_dataset()
    dataset_val = data_val.create_dataset()
    dataset_test = data_test.create_dataset()'''
    
    dataset_visualize = data_visualize.create_dataset()

    return dataset_visualize #dataset_train, dataset_val, dataset_test


def main():
    '''physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)'''

    #dataset_train, dataset_val, dataset_test = construct_datasets(CONSTANTS)
    dataset_visualize = construct_datasets(CONSTANTS)
    
    model = tf.keras.models.load_model('checkpoints/training_72/')

    '''model = NextSequencePredictor()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mape', optimizer=optimizer)
    model.fit(dataset_train, epochs=800, callbacks=[tensorboard_cb('tensorboard/training_73'),
    lr_scheduler(), checkpoints('checkpoints/training_73/')],
    validation_data = dataset_val)'''
    
    preds = model.predict(dataset_visualize)
    fig = plt.figure(figsize=(6, 6))
    columns = 2
    rows = 2
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
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    #plt.suptitle(f'Prediction of next {i} frames', y=0.65)
    plt.savefig('prediction.png')

    # model.evaluate(dataset_val, verbose=2)
    # model.save_weights('saved_weights/') # , save_format='h5')
    # model.save('saved_models/remote_3')  


if __name__ == "__main__":
    main()
