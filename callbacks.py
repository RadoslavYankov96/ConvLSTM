import tensorflow as tf
from keras import callbacks
import os

# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")


def tensorboard_cb(log_dir):
    tb_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=False,
    )
    return tb_callback


def lr_scheduler():
    def step_scheduler(epoch, lr):
        min_lr = 0.0001
        if epoch < 10:
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
        metric='val_mae',
        verbose=1,
        mode='min',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=True,
    )
    return checkpoint


def early_stopping():
    stopper = callbacks.EarlyStopping(
        monitor='val_mae',
        min_delta=0,
        patience=10,
        verbose=1,
    )
    return stopper




