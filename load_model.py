import keras
from prepare_dataset import ImageSequenceDataset

TEST_PATH = "C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\test\\"
IMG_DIMS = (512, 640, 1)

batch_size = 1
sequence_length = 2
starting_index = 6 - sequence_length

model = keras.models.load_model('saved_models/')
data_test = ImageSequenceDataset(TEST_PATH, sequence_length, batch_size, starting_index, IMG_DIMS)
dataset_test = data_test.create_dataset()

model.evaluate(dataset_test, verbose=2)
