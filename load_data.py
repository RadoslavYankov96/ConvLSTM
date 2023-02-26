from models import NextSequencePredictor
from prepare_dataset import ImageSequenceDataset
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_train = ImageSequenceDataset("C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\train\\",
                                  2, 1, 4, (512, 640, 1))
data_test = ImageSequenceDataset("C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\test\\",
                                 2, 1, 4, (512, 640, 1))

dataset_train = data_train.create_dataset()
dataset_test = data_test.create_dataset()
model = NextSequencePredictor()

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(dataset_train, epochs=15)
model.evaluate(dataset_test, verbose=2)
