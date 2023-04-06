import os
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import math


# FIRST_INDEX = 4
def dataset_histogram(data_root, first_index):
    hist = {}
    for target in os.listdir(data_root):
        data_path = os.path.join(data_root, target)
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            with h5.File(file_path, 'r') as experiment:
                keys = list(experiment.keys())[first_index:]
                for key in keys:
                    img = np.array(experiment[key], dtype=np.float32)
                    unique, counts = np.unique(img, return_counts=True)
                    img_dict = dict(zip(unique, counts))
                    hist = {k: hist.get(k, 0) + img_dict.get(k, 0) for k in set(hist) | set(img_dict)}
    return hist


def plot_histogram(histogram):
    plt.bar(histogram.keys(), histogram.values(), color='g', width=.0001, align='center')


def get_mean_std(histogram):
    pixel_values = np.array(list(hist.keys()))
    counts = np.array(list(hist.values()))
    total = np.sum(np.multiply(pixel_values, counts))
    avg = total/sum(counts)
    sum_of_deviation = 0
    for i, value in enumerate(pixel_values):
        sum_of_deviation += counts[i]*(value-avg)**2
    var = sum_of_deviation / sum(counts)
    sd = math.sqrt(var)


    return avg, sd


if __name__ == "__main__":
    hist = dataset_histogram("dataset", 4)
    mean, std = get_mean_std(hist)
    print(mean, std)
    plt.axvline(x=mean, color='r', label='mean')
    plt.axvline(x=mean-2*std, color='b', label='2 std left')
    plt.axvline(x=mean+2*std, color='b', label='2 std right')
    plot_histogram(hist)
    plt.legend()
    plt.show()
