import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math


def gb_homogeneity_score(img_dir):
    files = os.listdir(img_dir)
    img = np.load(os.path.join(img_dir, files[-2]))
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
    unique, counts = np.unique(array, return_counts=True)
    mean = np.sum(np.multiply(unique, counts))/np.sum(counts)
    sum_of_deviations = 0
    for i, value in enumerate(unique):
        sum_of_deviations += counts[i] * (value - mean) ** 2
    var = sum_of_deviations/np.sum(counts)
    std = math.sqrt(var)

    print(f"mean = {mean}")
    print(f"std = {std}")
    return mean, std


def statistical_homogeneity_score_from_img(img_dir):
    files = os.listdir(img_dir)
    img = np.load(os.path.join(img_dir, files[-1]))
    unique, counts = np.unique(img, return_counts=True)
    mean = np.sum(np.multiply(unique, counts))/np.sum(counts)
    sum_of_deviations = 0
    for i, value in enumerate(unique):
        sum_of_deviations += counts[i] * (value - mean) ** 2
    var = sum_of_deviations/np.sum(counts)
    std = math.sqrt(var)

    print(f"mean = {mean}")
    print(f"std = {std}")
    return mean, std


if __name__ == "__main__":
    print(f"gradient-based score: {gb_homogeneity_score('images')}")
    mean, std = statistical_homogeneity_score("images")
    # print(np.unique(counts))
    '''plt.bar(unique, counts, color='g', width=.8,  align='center')
    plt.axvline(mean, color='r', label="mean")
    plt.axvline(mean - std, color='b', label="std left")
    plt.axvline(mean + std, color='b', label="std right")'''
    #plt.show()
