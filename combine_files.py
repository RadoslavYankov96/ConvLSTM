import os
import h5py
import glob
from pathlib import Path

data_path = Path("C:\\Users\\rados\\Desktop\\studies\\thesis\\code\\DatasetConstruction\\"
                 "some_data")


for sequence in os.listdir(data_path):
    experiment_path = os.path.join(data_path, sequence)
    os.chdir(f"{data_path}\\dataset")
    with h5py.File(f"{sequence}.h5", mode='w') as h5fw:
        os.chdir(experiment_path)
        for i, h5name in enumerate(glob.glob('01022023*.h5')):
            h5old = h5py.File(h5name, 'r')
            data = h5old["D2_dataNorm"]
            h5fw.create_dataset(f"frame {str(i + 1).zfill(2)}", data=data)
