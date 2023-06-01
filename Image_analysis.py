# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:52:02 2022

@authors: al9140, ueuua
"""

import os
import glob
import time
import h5py
import numpy as np
import skimage.measure
from matplotlib import pyplot as plt


def steady_state_check(latest, previous, threshold, percentage_differences, num_rel_pixels):
    difference = latest - previous
    difference = np.absolute(difference)
    percentage_error = np.divide(difference, previous)
    percentage_error = skimage.measure.block_reduce(percentage_error, (16, 16), np.max)
    mean_error = np.mean(percentage_error)
    print('mean_error:', mean_error)
    histogram, edges = np.histogram(percentage_error, bins=10)
    percentage_differences.append(mean_error)
    deviating_pixels = np.count_nonzero(percentage_error > 0.03)
    size = percentage_error.size
    num_rel_pixels.append(deviating_pixels / size)
    print('Deviating pixels: ', deviating_pixels/size)
    not_steady_state = mean_error > threshold
    return not_steady_state, histogram, edges


def image_evaluation(input_files, latest_files, threshold, percentage_differences, num_rel_pixels):
    not_steady_state = True
    histogram = []
    edges = []
    list_of_files = glob.glob(input_files)
    if not list_of_files:
        pass
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = (os.path.normpath(latest_file))

        print(latest_file)
        # calling the converter function from itsimageanalyzer

        res = h5py.File(latest_file)
        img = res["D2_dataNorm"]
        img = np.array(img)
        latest_files.append(img)

        if len(latest_files) >= 2:
            latest_files = latest_files[-2:]
            latest = latest_files[-1]
            previous = latest_files[-2]
            if (previous != latest).any():
                not_steady_state, histogram, edges = steady_state_check(latest, previous, threshold,
                                                                        percentage_differences, num_rel_pixels)

    return not_steady_state, histogram, edges


def image_evaluation_labview(input_files, threshold):
    list_of_files = filter(os.path.isfile, glob.glob(input_files + '*'))
    latest_files = sorted(list_of_files, key=os.path.getmtime)
    if not latest_files:
        return True
    if len(latest_files) == 1:
        return True
    else:
        latest_files = latest_files[-2:]
        previous = latest_files[0]
        latest = latest_files[1]

        pre_dict = h5py.File(previous)
        lat_dict = h5py.File(latest)
        pre_img = pre_dict['D2_dataNorm']
        lat_img = lat_dict['D2_dataNorm']
        pre_img = np.array(pre_img)
        lat_img = np.array(lat_img)
        if previous != latest:
            difference = lat_img - pre_img
            difference = np.absolute(difference)
            percentage_error = np.divide(difference, pre_img)
            percentage_error = skimage.measure.block_reduce(percentage_error, (16, 16), np.max)
            mean_error = np.mean(percentage_error)
            if mean_error > threshold:
                return True
        
            return False


def main():
    # INPUT
    # file to convert. Must be the sfmov file with the remaining 4-5 files of same image in the same directory
    input_files = "H:/public/images/GP/energy_lab/second_recording/*.h5"
    threshold = 0.01
    not_steady_state = True
    latest_files = []
    percentage_differences = []
    num_rel_pixels = []
    outputs = [percentage_differences, num_rel_pixels]
    start_time = time.time()

    while not_steady_state:

        not_steady_state, histogram, edges = image_evaluation(input_files, latest_files, threshold, *outputs)

        '''if not not_steady_state:
            end_time = time.time()
            print(end_time - start_time)

            plt.stairs(histogram, edges)
            plt.show()

            plt.plot(percentage_differences, color='blue', marker='o', mfc='red')  # plot the data
            plt.xticks(range(0, len(percentage_differences) + 1, 1))  # set the tick frequency on x-axis

            plt.ylabel('Temperature difference between images [%]')  # set the label for y-axis
            plt.xlabel('img_index')  # set the label for x-axis
            plt.title("Temperature Field Evolution")  # set the title of the graph
            plt.show()  # display the graph

            plt.plot(num_rel_pixels, color='blue', marker='o', mfc='red')  # plot the data
            plt.xticks(range(0, len(num_rel_pixels) + 1, 1))  # set the tick frequency on x-axis

            plt.ylabel('Num of pixels with relevant Temperature deviation [%]')  # set the label for y-axis
            plt.xlabel('img_index')  # set the label for x-axis
            plt.title("Percentage of relevant pixels after 16x16 maxpool ")  # set the title of the graph
            plt.show()  # display the graph'''

        time.sleep(30)


if __name__ == "__main__":
    main()
