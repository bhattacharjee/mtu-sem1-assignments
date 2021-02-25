#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import math

from Lab_MV_04_Interest_points import *

def load_image_and_convert_to_grayscale()->np.ndarray:
    """
    Open the image and convert to grayscale
    """
    image = cv2.imread('Assignment_MV_1_image.png')
    #image = cv2.imread('sobx.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image_gray = image_gray.astype(float)
    image_gray = np.float32(image_gray)
    return image_gray

def display_image(i:np.ndarray)->None:
    """
    Display an image and wait for an input
    """
    cv2.imshow('gray', i)
    cv2.waitKey(0)

def display_images_meshgrid(images:list, gray=True, title_arr=None)->None:
    """
    Display multiple images side by side
    Images are in form of a list of np.ndarray
    """
    n_images = len(images)
    rows = math.ceil(math.sqrt(n_images))
    cols = math.ceil(n_images / rows)
    fig, ax = plt.subplots(rows, cols)
    index = -1
    for r in range(rows):
        for c in range(cols):
            index += 1
            if index >= len(images):
                plt.show()
                return
            image = images[index]
            arr = np.asarray(image)
            if rows >= 2 and cols >= 2:
                tax = ax[r][c]
            elif rows >= 2 or cols >= 2:
                tax = ax[index-1]
            else:
                tax = ax
            if gray:
                tax.imshow(arr, cmap='gray', vmin=0, vmax=255)
            else:
                tax.imshow(arr)
            if title_arr:
                tax.title.set_text(title_arr[index])
    plt.show()
    return

def get_gaussian_smoothing_kernel(size:int, sigma:float)->np.ndarray:
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    x = x - len(x) / 2
    y = y - len(y) / 2
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)

    # Ideally the area under a gaussian must be 1, but in our case
    # When sigma is large and the size is small, the gaussian will
    # not be covered in the complete window, and so the image will
    # appear to lose brightness when convolved with this gaussian
    # The fix is to normalize to 1
    kernel = kernel / np.sum(kernel)
    return kernel

def task_1b()->tuple:
    titles = []
    images = []
    gaussians = []
    image = load_image_and_convert_to_grayscale()
    images.append(image)
    titles.append("base")
    for k in range(12):
        sigma = math.pow(2, k/2)
        g = get_gaussian_smoothing_kernel(50, sigma)
        im = image.copy()
        im = cv2.filter2D(im, -1, g)
        images.append(im)
        gaussians.append(g)
        titles.append(f"sigma={sigma}")
    display_images_meshgrid(gaussians, gray=False)
    display_images_meshgrid(images, True, titles)
    return images, titles

def task_2a()->tuple:
    titles = []
    images = []
    image = load_image_and_convert_to_grayscale()
    images.append(image)
    titles.append("base")
    for k in range(12):
        sigma = math.pow(2, k/2)
        sigma2 = math.pow(2, (k+1)/2)
        g = get_gaussian_smoothing_kernel(200, sigma)
        g2 = get_gaussian_smoothing_kernel(200, sigma2)
        im = image.copy()
        im = cv2.filter2D(im, -1, g)
        im2 = image.copy()
        im2 = cv2.filter2D(im2, -1, g2)
        # Since there are negative values, we must offset by 128
        im = im - im2 + 128
        images.append(im)
        titles.append(f"dog {sigma} - {sigma2}")
    display_images_meshgrid(images, False, titles)
    return images, titles
        
def task_2b()->None:
    images, titles = task_2a()
    for i in range(1, len(images)):
        im = images[i]
        thresh = np.min(im) + (np.max(im) - np.min(im))/2
        im[im <= thresh] = 0
        im[im > thresh] = 255
        images[i] = im
    display_images_meshgrid(images, True, titles)


task_2b()
