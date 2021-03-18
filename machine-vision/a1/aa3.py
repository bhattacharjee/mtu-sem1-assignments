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

GAUSSIAN_KERNEL_SIZE_MULTIPLIER = 3
def get_gaussian_smoothing_kernel_internal(size:int, sigma:float):
    p_x, p_y = np.meshgrid(\
           np.arange(-GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma, \
                GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma),\
           np.arange(-GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma,\
                GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma))
    return np.exp(-(p_x**2 + p_y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)    

def get_gaussian_smoothing_kernel(sigma:float)->np.ndarray:
    return get_gaussian_smoothing_kernel_internal(math.ceil(6 * sigma), sigma)

def get_twelve_smoothed_images(image)->tuple:
    sigma_arr = []
    images = []
    for k in range(12):
        sigma = (2 ** (k/2))
        sigma_arr.append(sigma)
        g = get_gaussian_smoothing_kernel(sigma)
        im = cv2.filter2D(image.copy(), -1, g)
        images.append(im)
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
        if ((pixel_value > T) and
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

"""
Get gaussian derivatives of a list of images in different sigma scales
The output is a list, each element of the list is a tuple.
The first item in the tuple is gx, and the second item in the tuple
is gy
"""
def get_derivative_of_gaussian(gaussian_images:list, sigma_list)->list:
    retval = []
    dx = np.array([[1, 0, -1]])
    dy = dx.T
    #dx = cv2.flip(dx, -1)
    #dy = cv2.flip(dy, -1)
    for image in gaussian_images:
        gx = cv2.filter2D(image.copy(), -1, dx)
        gy = cv2.filter2D(image.copy(), -1, dy)
        retval.append((gx, gy, ))
    return retval

@lru_cache
def get_weight(q, r, sigma):
    wqr = np.exp( -(q**2 + r**2) / (9 * sigma**2 / 2 ))
    wqr = wqr / (9 * np.pi * sigma**2 / 2)
    return wqr

def get_augmented_point_with_direction(im, x, y, sigma, m_gx, m_gy, m_theta,\
                                        m_magnitude):
    hist = [0.0] * 36
    for k in range(-3, 4):
        for j in range(-3, 4):
            q = round(3/2 * k * sigma)
            r = round(3/2 * j * sigma)
            xq = x + q
            yr = y + r
            mqr = m_magnitude[xq, yr]
            tqr = (m_theta[xq, yr]  + np.pi) * 180 / np.pi
            wqr = get_weight(q, r, sigma)
            i = int(math.floor(tqr / 10))
            hist[i] += (wqr * mqr)
    thetahat = (2 * np.pi / 36) * (0.5 + np.argmax(np.array(hist))) - np.pi
    thetahat = (2*np.pi/36)*(hist[np.argmax(hist)]+1/2)
    return (x, y, sigma, thetahat, )

def show_derivative_of_gaussian_images(images):
    gxa = []
    gya = []
    for dog in images:
        gx = dog[0]
        gy = dog[1]
        gxa.append(gx)
        gya.append(gy)
    display_images_meshgrid(gxa, True, None)
    display_images_meshgrid(gya, True, None)

def task_3():
    image = load_image_and_convert_to_grayscale()
    save_image = load_original_image()
    images, sigma_arr = get_twelve_smoothed_images(image)
    key_points = task_2()
    dog_images = get_derivative_of_gaussian(images, sigma_arr)
    magnitudes = []
    theta_array = []
    sigma_cross_ref = {}

    #show_derivative_of_gaussian_images(dog_images)
    #plt.show()

    # Calculate the magnitude and angle for all points
    # We will select the points that matter to us later
    for im, sigma, (gx, gy) in zip(images, sigma_arr, dog_images):
        magnitude = np.sqrt(np.square(gx) + np.square(gy))
        magnitudes.append(magnitude)
        theta = np.arctan2(gy, gx)
        theta_array.append(theta)
        sigma_cross_ref[sigma] = {
                "gx": gx,
                "gy": gy,
                "gx2": np.square(gx.copy()),
                "gy2": np.square(gy.copy()),
                "image": im,
                "theta": theta,
                "magnitude": magnitude
                }

    augmented_points = []
    exception_count = 0
    for point in key_points:
        try:
            x, y, sigma = point
            augmented_points.append(get_augmented_point_with_direction(\
                    sigma_cross_ref[sigma]["image"],\
                    x,\
                    y,\
                    sigma,\
                    sigma_cross_ref[sigma]["gx"],\
                    sigma_cross_ref[sigma]["gy"],\
                    sigma_cross_ref[sigma]["theta"],\
                    sigma_cross_ref[sigma]["magnitude"]))
        except:
            exception_count += 1
            #print(f"got exception {exception_count}")
            
    save_image = load_original_image()
    for point in augmented_points:
        x, y, sigma, angle = point
        x = int(x)
        y = int(y)
        radius = math.floor(3 * sigma)
        cv2.circle(save_image, (y, x,), radius, (0,255,0,), 1)
        x1 = int(round(x + radius * math.cos(angle)))
        y1 = int(round(y + radius * math.sin(angle)))
        cv2.line(save_image, (y, x, ), (y1, x1, ), (0,0,255,), 2)
    cv2.imshow("result", save_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return augmented_points, sigma_cross_ref

@lru_cache
def get_weight2(s, t, sigma):
    return np.exp(-(s**2 + t**2) / (81 * sigma**2 /2)) / (81 * np.pi * sigma**2 / 2)

def task_4():
    augmented_points, sigma_crf = task_3()
    for point in augmented_points:
        try:
            x, y, sigma, theta_hat = point
            theta_arr = sigma_crf[sigma]["theta"]
            magnitude_arr = sigma_crf[sigma]["magnitude"]
            all_hist = []
            for i in range(-2, 2):
                for j in range(-2, 2):
                    hist = [0.0] * 8
                    for k1 in range(0, 4):
                        for k2 in range(0, 4):
                            s = int(round(9/16 * (k1 + 0.5) * sigma))
                            t = int(round(9/16 * (k2 + 0.5) * sigma))
                            wst = get_weight2(s, t, sigma)
                            xs = x + s
                            yt = y + t
                            mst = magnitude_arr[xs, yt]
                            theta_st = theta_arr[xs, yt] - theta_hat
                            ind = int(math.floor(theta_st / (2 * np.pi / 8)))
                            hist[ind] += (wst * mst)
                    all_hist.extend(hist)
            all_hist = np.array(all_hist)
            all_hist = all_hist / np.sqrt(np.sum(np.square(all_hist)))
            all_hist[all_hist > 0.2] = 0.2
            #print(all_hist)
        except:
            pass


task_4()

