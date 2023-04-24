import os
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import math
import glob
import re 


# FIRST_INDEX = 4
def dataset_histogram(data_root, first_index):
    hist = {}
    for target in os.listdir(data_root):
        data_path = os.path.join(data_root, target)
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            with h5.File(file_path, 'r') as experiment:
                keys = list(experiment.keys())
                keys = sorted(keys, key=lambda s: int(re.search(r'\d+', s).group()))
                keys = keys[first_index:]
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
    
    
def create_csv(data_path, work_dir):
    os.chdir(data_path)
    files = glob.glob('*.h5')
    print(files)
    os.chdir(work_dir)
    np.savetxt('dataset_info.csv',
               files,
               delimiter="_",
               fmt='%s'
               )

if __name__ == "__main__":
    hist = dataset_histogram("/home/itsnas/ueuua/BA/dataset", 4)
    mean, std = get_mean_std(hist)
    print(mean, std)
    
    os.chdir("/home/itsnas/ueuua/BA/visualizations")
    
    plt.axvline(x=mean, color='r', label='mean')
    plt.axvline(x=mean-2*std, color='b', label='2 std left')
    plt.axvline(x=mean+2*std, color='b', label='2 std right')
    plot_histogram(hist)
    plt.xlabel('normalized temperature', loc='right')
    plt.ylabel('# of pixels', loc='top')
    plt.title("Temperature Distribution of Pixels within the Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig('dataset_summary.png')
    
    '''data_path = '/home/itsnas/ueuua/BA/dataset/train'
    work_dir = '/home/itsnas/ueuua/BA/dataset'
    create_csv(data_path, work_dir)'''
