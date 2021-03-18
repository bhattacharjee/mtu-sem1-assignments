#!/usr/bin/env python3
# coding: utf-8

from functools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import math

from Lab_MV_04_Interest_points import *

GAUSSIAN_ADD = 0
THRESHOLD = 10

def load_original_image()->np.ndarray:
    return cv2.imread('Assignment_MV_1_image.png')

def load_image_and_convert_to_grayscale()->np.ndarray:
    """
    Open the image and convert to grayscale
    """
    image = load_original_image()
    #image = cv2.imread('sobx.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                #plt.show()
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
                tax.imshow(arr, cmap='gray')#, vmin=0, vmax=255)
            else:
                tax.imshow(arr)
            if title_arr:
                tax.title.set_text(title_arr[index])
    #plt.show()
    return

def get_gaussian_smoothing_kernel_internal(size:int, sigma:float)->np.ndarray:
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
    #kernel = kernel / np.sum(kernel)
    return kernel

KERNEL_SIZE = 3
# Copied from gyani
def get_gaussian_smoothing_kernel_internal(size:int, sigma:float):
    point_x, point_y = np.meshgrid(np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma),
                       np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma))
    return np.exp(-(point_x**2 + point_y**2)/(2*sigma**2))/(2*np.pi*sigma**2)    

def get_gaussian_smoothing_kernel(sigma:float)->np.ndarray:
    return get_gaussian_smoothing_kernel_internal(math.ceil(6 * sigma), sigma)

def get_smoothed_images(image, sigma_values, kernel_size=100)->list:
    smoothed_images = []
    for sigma in sigma_values:
        g = get_gaussian_smoothing_kernel(sigma)
        im = cv2.filter2D(image.copy(), -1, g)
        smoothed_images.append(im)
    return smoothed_images
        
def get_twelve_smoothed_images(image)->tuple:
    sigma_arr = []
    for k in range(12):
        #sigma_arr.append(math.pow(2, k/2))
        sigma_arr.append(2 ** (k/2))
    images = get_smoothed_images(image, sigma_arr)
    return images, sigma_arr

def get_difference_of_gaussian_images(images, sigma_arr)->list:
    dog_arr = []
    for i in range(len(images)-1):
        i1, i2 = images[i], images[i+1]
        s1, s2 = sigma_arr[i], sigma_arr[i+1]
        dog = i2 - i1 + GAUSSIAN_ADD
        dog_arr.append({"dog": dog, "sigma1": s1, "sigma2": s2, })
    return dog_arr

def is_maximal_pixel(metric, x, y, pixel_value, T, check_same_level=True):
    try:
        if ((pixel_value >T) and
            (pixel_value > metric[x-1,y-1]) and
            (pixel_value > metric[x-1,y])   and
            (pixel_value > metric[x-1,y+1]) and
            (pixel_value > metric[x,y-1])   and
            (pixel_value > metric[x,y+1])   and
            (pixel_value > metric[x+1,y-1]) and
            (pixel_value > metric[x+1,y])   and
            (pixel_value > metric[x+1,y+1])):
                if check_same_level and pixel_value > metric[x, y]:
                    return True
                elif not check_same_level:
                    return True
                else:
                    return False
        else:
            return False
    except:
        return False
    return False

def get_non_maxima_suppression_pixels(image, lower, higher, T, sigma):
    points = []
    for x in range(1,len(image)-1):
        for y in range(1,len(image[0])-1):
            m1, m2, m3 = True, True, True 
            if lower is not None:
                m1 = is_maximal_pixel(lower, x, y, image[x, y], T, True)
            if not m1:
                continue
            if higher is not None:
                m3 = is_maximal_pixel(higher, x, y, image[x, y], T, True)
            if not m3:
                continue
            m2 = is_maximal_pixel(image, x, y, image[x, y], T, False)
            if m1 and m2 and m3:
                points.append((x, y, sigma, ))
    return points

def get_all_non_maxima_suppression_pixels(dog_arr, T):
    points = []
    #for i in range(2):
    for i in range(len(dog_arr)):
        print(f"processing {i}...")
        lower = higher = image = None
        x = dog_arr[i]
        image = x["dog"]
        sigma = x["sigma1"]
        if i - 1 >= 0:
            lower = dog_arr[i - 1]["dog"]
        else:
            continue
        if i + 1 < len(dog_arr):
            higher = dog_arr[i + 1]["dog"]
        else:
            continue
        p = get_non_maxima_suppression_pixels(image, lower, higher, T, sigma)
        [points.append(x) for x in p]
        print(f"{sigma} {i} - {len(points)}")
    return points

def task_1b():
    images, sigma_arr = get_twelve_smoothed_images(load_image_and_convert_to_grayscale())
    print(sigma_arr)
    display_images_meshgrid(images)
    gaussian_kernels = []
    for s in sigma_arr:
        gaussian_kernels.append(get_gaussian_smoothing_kernel(s))
    display_images_meshgrid(gaussian_kernels, gray=False)

def task_2()->list:
    image = load_image_and_convert_to_grayscale()
    save_image = load_original_image()
    images, sigma_arr = get_twelve_smoothed_images(image)

    include_base_image = False
    if include_base_image:
        new_images = [image.copy()]
        [new_images.append(x) for x in images]
        new_sigma_arr = [0.5]
        [new_sigma_arr.append(s) for s in sigma_arr]
    else:
        new_images = images
        new_sigma_arr = sigma_arr

    dog_arr = get_difference_of_gaussian_images(new_images, new_sigma_arr)
    display = []
    display_text = []
    for x in dog_arr:
        dog, s1, s2 = x["dog"], x["sigma1"], x["sigma2"]
        display_text.append(f"{s1} - {s2}")
        display.append(dog)
    print("displaying dog")
    display_images_meshgrid(display, True, display_text)
    points = get_all_non_maxima_suppression_pixels(dog_arr, GAUSSIAN_ADD+THRESHOLD)
    print(f"Number of points found = {len(points)}")
    for point in points:
        x, y, sigma = point
        x = int(x)
        y = int(y)
        radius = math.floor(3 * sigma)
        cv2.circle(save_image, (y, x,), radius, (0,255,0), 1)
    cv2.imshow("result", save_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

def get_derivative_of_gaussian_both_directions_combined(gaussian_images:list, sigma_list)->list:
    images = []
    dx = np.array([[1, 0, -1], ])
    dy = np.array([[1, 0, -1], ]).T
    for image, sigma in zip(gaussian_images, sigma_list):
        image = image.copy()
        image = cv2.filter2D(image, -1, cv2.flip(dx, -1))
        image = cv2.filter2D(image, -1, cv2.flip(dy, -1))
        image = image * (sigma * sigma)
        image = image + GAUSSIAN_ADD
        images.append(image)
    return images


"""
Get gaussian derivatives of a list of images in different sigma scales
The output is a list, each element of the list is a tuple.
The first item in the tuple is gx, and the second item in the tuple
is gy
"""
def get_derivative_of_gaussian(gaussian_images:list, sigma_list)->list:
    retval = []
    dx = np.array([[1, 0, -1], ])
    dy = np.array([[1, 0, -1], ]).T
    for image, sigma in zip(gaussian_images, sigma_list):
        i = image.copy()
        gx = cv2.filter2D(i, -1, cv2.flip(dx, -1))
        i = image.copy()
        gy = cv2.filter2D(i, -1, cv2.flip(dy, -1))
        retval.append((gx, gy, ))
    return retval


        
@lru_cache
def get_interpolation_matrix(sigma)->np.ndarray:
    size = int(3/2 * 3 * sigma) - int(3/2 * -3 * sigma) + 2
    retval = np.zeros((size, size))
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = int((size / 2) + (3 / 2 * i * sigma))
            y = int((size / 2) + (3 / 2 * j * sigma))
            retval[x,y] = 1
    return retval, round(size/2), round(size/2)

@lru_cache
def get_weighted_gaussian_matrix(sigma)->np.ndarray:
    size = round(3/2 * 3 * sigma) - round(3/2 * -3 * sigma) + 2
    retval = np.zeros((size, size))
    for i in range(-3, 4):
        for j in range(-3, 4):
            exp_numerator = -(i ** 2 + j ** 2)
            exp_denominator = (9 * sigma * sigma / 2)
            denominator = 9 * np.pi * sigma * sigma / 2
            x = int(round(size / 2) + (3 / 2 * i * sigma))
            y = int(round(size / 2) + (3 / 2 * j * sigma))
            retval[x,y] = np.exp(exp_numerator / exp_denominator) / denominator
    return retval, round(size/2), round(size/2)



def task_3():
    image = load_image_and_convert_to_grayscale()
    save_image = load_original_image()
    images, sigma_arr = get_twelve_smoothed_images(image)
    key_points = task_2()
    dog_images = get_derivative_of_gaussian(images, sigma_arr)
    magnitudes = []
    theta_array = []
    sigma_cross_ref = {}

    # Calculate the magnitude and angle for all points
    # We will select the points that matter to us later
    for im, sigma, dog in zip(images, sigma_arr, dog_images):
        gx, gy = dog[0], dog[1]
        magnitude = np.sqrt(np.square(gx) + np.square(gy))
        magnitudes.append(magnitude)
        theta = np.arctan2(gy, gx)
        theta_array.append(theta)
        sigma_cross_ref[sigma] = {
                "image": im,
                "gx": gx,
                "gy": gy,
                "magnitude": magnitude,
                "theta": theta
                }

    augmented_points = []
    for point in key_points:
        x, y, sigma = point
        int_matrix, int_matrix_centre_x, int_matrix_centre_y = get_interpolation_matrix(sigma)
        wt_matrix, wt_matrix_centre_x, wt_matrix_centre_y = get_weighted_gaussian_matrix(sigma)
        int_matrix_size = int_matrix.shape[0]
        low_x = round(x - int_matrix_size / 2)
        low_y = round(y - int_matrix_size / 2)

        """
        Instead of a lot of boundary condition checks, for array indices
        just use exception handling to keep it clean
        """
        try:
            high_x = low_x + int_matrix_size
            high_y = low_y + int_matrix_size

            gx = sigma_cross_ref[sigma]["gx"]
            gy = sigma_cross_ref[sigma]["gy"]
            im = sigma_cross_ref[sigma]["image"]
            th = sigma_cross_ref[sigma]["theta"]
            mg = sigma_cross_ref[sigma]["magnitude"]

            th_slice = th[low_x:high_x, low_y:high_y]
            mg_slice = mg[low_x:high_x, low_y:high_y]

            wt_mg = mg_slice * wt_matrix
            th_mg = th_slice * int_matrix

            hist = [0] * 36
            for i in range(th_mg.shape[0]):
                for j in range(th_mg.shape[1]):
                    if int_matrix[i,j] != 0:
                        angle = th_mg[i,j]
                        thebin = math.floor((angle + np.pi) / (2 * np.pi / 36))
                        hist[thebin] += wt_mg[i,j]

            out_angle = 2 * np.pi / 36 * (0.5 + np.argmax(np.array(hist))) - np.pi
            augmented_points.append((x, y, sigma, out_angle, ))
        except:
            print("exception, continuing")
            pass
        

    save_image = load_original_image()
    for point in augmented_points:
        x, y, sigma, angle = point
        x = int(x)
        y = int(y)
        radius = math.floor(3 * sigma)
        cv2.circle(save_image, (y, x,), radius, (0,255,0,), 1)
        x1 = int(round(x + radius * np.cos(angle)))
        y1 = int(round(y + radius * np.sin(angle)))
        cv2.line(save_image, (y, x, ), (y1, x1, ), (0,0,255,), 2)
    cv2.imshow("result", save_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


task_1b()
task_3()

