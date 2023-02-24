from models import SequencePredictor
from prepare_data import ImageSequenceDataset

data = ImageSequenceDataset("C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\ConvLSTM\\dataset\\",
                            2, 4, 4, (520, 640, 1))

dataset = data.create_dataset()
model = SequencePredictor(5, 3, [16, 32, 32], 2, [2, 520, 640, 1], (1, 2, 2))
model.build_model()
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(dataset, epochs=10)
