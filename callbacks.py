import tensorflow as tf
from tensorflow.keras import callbacks
import os

# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")


class ShuffleCallback(callbacks.Callback):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 40:
            self.dataset.shuffle(100)
        return self.dataset
        

def shuffle_cb(dataset):
    ds_shuffle = ShuffleCallback(dataset=dataset)
    return ds_shuffle
            

def tensorboard_cb(log_dir):
    tb_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=False,
    )
    return tb_callback


def lr_scheduler():
    def step_scheduler(epoch, lr):
        min_lr = 0.00001
        if epoch < 20:
            return lr
        elif lr > min_lr:
            return lr * 0.975
        else:
            return lr

    lr_schedule = callbacks.LearningRateScheduler(
        schedule=step_scheduler, verbose=1
    )
    return lr_schedule


def checkpoints(chp_dir):
    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(chp_dir),
        metric='val_loss',
        verbose=1,
        mode='min',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=True,
    )
    return checkpoint


def early_stopping():
    stopper = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        mode='min',
        patience=50,
        verbose=1,
    )
    return stopper




