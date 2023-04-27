import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math


def homogeneity_evaluation(array):
    array = array[:, -1, :, :, :]
    array = np.squeeze(array)
    mean = np.mean(array)
    std = np.std(array)
    hot_pixels = array - mean
    hot_score = np.sum(hot_pixels, where=hot_pixels>0)
    return std, hot_score


def gb_homogeneity_score(img_dir):
    files = os.listdir(img_dir)
    img = np.load(os.path.join(img_dir, files[-4:-2]))
    img = np.reshape(img, (512, 640))
    sobel_x = np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32)
    sobel_y = np.abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32)
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y), dtype=np.float32)
    # sobel_visualize = cv2.convertScaleAbs(sobel)
    plt.imshow(sobel, cmap="gray", vmin=0, vmax=30)
    plt.show()
    grad_score = np.sum(sobel)
    return grad_score


def statistical_homogeneity_score(array):
    array = array[:, -1, :, :, :]
    array = np.squeeze(array)
    unique, counts = np.unique(array, return_counts=True)
    mean = np.sum(np.multiply(unique, counts))/np.sum(counts)
    sum_of_deviations = 0
    for i, value in enumerate(unique):
        sum_of_deviations += counts[i] * (value - mean) ** 2
    var = sum_of_deviations/np.sum(counts)
    std = math.sqrt(var)
    
    return mean, std
    
    
def hot_outliers_score(array):
    array = array[:, -1, :, :, :]
    array = np.squeeze(array)
    # array = np.multiply(array, 255)
    unique, counts = np.unique(array, return_counts=True)
    mean = np.sum(np.multiply(unique, counts)) / np.sum(counts)
    array = array - mean
    score = np.sum(array, where=array > 0)

    return score
    
    
def hot_outliers_score_from_img(img_dir):
    files = os.listdir(img_dir)
    print(files[-1])
    img = np.load(os.path.join(img_dir, files[-1]))
    unique, counts = np.unique(img, return_counts=True)
    mean = np.sum(np.multiply(unique, counts)) / np.sum(counts)
    diff = img - mean
    score = np.sum(diff, where=diff > 0)

    return score        


def statistical_homogeneity_score_from_img(img_dir):
    files = os.listdir(img_dir)
    print(files[-1])
    img = np.load(os.path.join(img_dir, files[-1]))
    unique, counts = np.unique(img, return_counts=True)
    mean = np.sum(np.multiply(unique, counts))/np.sum(counts)
    sum_of_deviations = 0
    for i, value in enumerate(unique):
        sum_of_deviations += counts[i] * (value - mean) ** 2
    var = sum_of_deviations/np.sum(counts)
    std = math.sqrt(var)

    mean1 = np.mean(img)
    std1 = np.std(img)

    print(f"mean = {mean}, {mean1}")
    print(f"std = {std}, {std1}")
    return mean, std


if __name__ == "__main__":
    # print(f"gradient-based score: {gb_homogeneity_score('images')}")
    mean, std = statistical_homogeneity_score_from_img("images")
    # print(np.unique(counts))
    '''plt.bar(unique, counts, color='g', width=.8,  align='center')
    plt.axvline(mean, color='r', label="mean")
    plt.axvline(mean - std, color='b', label="std left")
    plt.axvline(mean + std, color='b', label="std right")'''
    #plt.show()
