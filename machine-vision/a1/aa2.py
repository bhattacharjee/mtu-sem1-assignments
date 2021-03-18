#!/usr/bin/env python3
# coding: utf-8

from functools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import math
import argparse

#from Lab_MV_04_Interest_points import *

# This is really not used
GAUSSIAN_ADD = 0

# The threshold value for non-maxima suppression
THRESHOLD = 10

# This will be overwritten by argument parser
g_filename = 'Assignment_MV_1_image.png'

"""
This function just loads teh image
"""
def load_original_image()->np.ndarray:
    global g_filename
    return cv2.imread(g_filename)

"""
This function loads the image and converts it to grayscale
"""
def load_image_and_convert_to_grayscale()->np.ndarray:
    """
    Open the image and convert to grayscale
    """
    image = load_original_image()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    return image_gray

"""
Display a single image using cv2.imshow
"""
def display_image(i:np.ndarray)->None:
    """
    Display an image and wait for an input
    """
    cv2.imshow('gray', i)
    cv2.waitKey(0)


"""
Given an array of images, show all the images in the same window.
There are additional arrays which contains the titles for each image
"""
def display_images_meshgrid(\
        images:list,\
        gray=True, \
        title_arr=None,\
        suptitle=None)->None:
    """
    Display multiple images side by side
    Images are in form of a list of np.ndarray
    """
    n_images = len(images)
    rows = math.ceil(math.sqrt(n_images))
    cols = math.ceil(n_images / rows)
    fig, ax = plt.subplots(rows, cols)
    if None != suptitle:
        fig.suptitle(suptitle)
    index = -1
    for r in range(rows):
        for c in range(cols):
            index += 1
            if index >= len(images):
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
    return

GAUSSIAN_KERNEL_SIZE_MULTIPLIER = 3
"""
Given a sigma, return a gaussian smoothing kernel of the appropriate size
"""
def get_gaussian_smoothing_kernel(sigma:float)->np.ndarray:
    p_x, p_y = np.meshgrid(\
           np.arange(-GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma, \
                GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma),\
           np.arange(-GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma,\
                GAUSSIAN_KERNEL_SIZE_MULTIPLIER * sigma))
    return np.exp(-(p_x**2 + p_y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)    

"""
Get 12 gaussian smoothing kernels with different sigmas
where sigma = (2 ** (k/2)), k = 0,1,...,11
"""
def get_twelve_smoothed_images(image:np.ndarray)->tuple:
    sigma_arr = []
    images = []
    for k in range(12):
        sigma = (2 ** (k/2))
        sigma_arr.append(sigma)
        g = get_gaussian_smoothing_kernel(sigma)
        im = cv2.filter2D(image.copy(), -1, g)
        images.append(im)
    return images, sigma_arr

"""
Get difference of gaussian images, with different sigma values
"""
def get_difference_of_gaussian_images(images:np.ndarray,\
        sigma_arr:list)->list:
    dog_arr = []
    for i in range(len(images)-1):
        i1, i2 = images[i], images[i+1]
        s1, s2 = sigma_arr[i], sigma_arr[i+1]
        dog = i2 - i1 + GAUSSIAN_ADD
        dog_arr.append({"dog": dog, "sigma1": s1, "sigma2": s2, })
    return dog_arr


"""
This function is called multiple times. It checks if a pixel is maximal
or not.
There is an additional parameter check_same_level
If this parameter is True, then it will check [x,y], if not, it will not
check [x,y]
This is because this will be called thrice for each layer,
1. once for sigma = 2**(k/2)
2. Once for sigma = 2**((k-1)/2)
3. Once for sigma = 2**((k+1)/2)

For 1, check_same_level=False and True otherwise
"""
def is_maximal_pixel(\
        metric:np.ndarray,\
        x:int,\
        y:int,\
        pixel_value:float,\
        T:int,\
        check_same_level:bool=True)->bool:
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

"""
For a particular level in scale-space, get all non-maxima suppressed pixels
at that level.
This is done by calling is_maximal_pixel for every pixel
"""
def get_non_maxima_suppression_pixels(\
        image:np.ndarray,\
        lower:np.ndarray,\
        higher:np.ndarray,\
        T:int,\
        sigma:list)->list:
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

"""
Get all keypoints for all levels in scale space
"""
def get_all_non_maxima_suppression_pixels(dog_arr:list, T:int)->list:
    points = []
    for i in range(len(dog_arr)):
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
        print(f"Finding keypoints: sigma {sigma:2.2f} - {len(points)} points")
    return points

"""
Gets twelve gaussian kernels in the sigma range
For each gaussian kernel, it applies it to the image
It then displays two things:
    1. The Gaussian kernel
    2. The image after application of the gaussian kernel (scale space)
"""
def task_1b():
    images, sigma_arr = get_twelve_smoothed_images(\
            load_image_and_convert_to_grayscale())
    title_arr = []
    for s in sigma_arr:
        title_arr.append(f"sigma = {s:2.2f}")
    display_images_meshgrid(images, \
            gray=True, title_arr=title_arr, suptitle="Gaussian Smoothed Image")
    gaussian_kernels = []
    for s in sigma_arr:
        gaussian_kernels.append(get_gaussian_smoothing_kernel(s))
    display_images_meshgrid(gaussian_kernels, \
            gray=False, title_arr=title_arr, suptitle="gaussian kernels")
    plt.show()

"""
1. Gets the scale space representation with the gaussian kernel at different
   sigma values applied to the image (same from task 1B)
2. Find the difference of Gaussians
3. Find key-points by using non-maxima suppression with a threshold 10
4. Visualize the key-points by drawing a circle with radius 3 * sigma
"""
def task_2()->list:
    image = load_image_and_convert_to_grayscale()
    save_image = load_original_image()
    images, sigma_arr = get_twelve_smoothed_images(image)

    # This variable controls whether the original image is considered in the
    # scale space or not.
    # For now we set it to false
    include_base_image = False

    if include_base_image:
        new_images = [image.copy()]
        [new_images.append(x) for x in images]
        new_sigma_arr = [0.5]
        [new_sigma_arr.append(s) for s in sigma_arr]
    else:
        new_images = images
        new_sigma_arr = sigma_arr

    # Get the difference of Gaussian by calling this function
    dog_arr = get_difference_of_gaussian_images(new_images, new_sigma_arr)

    # Now display all the difference of Gaussian images. Most of the code here
    # Is to get the right title appended to each dog
    display = []
    display_text = []
    for x in dog_arr:
        dog, s1, s2 = x["dog"], x["sigma1"], x["sigma2"]
        display_text.append(f"{s2} - {s1}")
        display.append(dog)
    display_images_meshgrid(display,\
            True, display_text, "Difference of Gaussian")
    plt.show()

    # Get all key points by non-maxima suppression
    points = get_all_non_maxima_suppression_pixels(dog_arr,\
                GAUSSIAN_ADD+THRESHOLD)
    print(f"Number of points found = {len(points)}")

    # For each of the key points, draw a circle
    for point in points:
        x, y, sigma = point
        x = int(x)
        y = int(y)
        radius = math.floor(3 * sigma)
        cv2.circle(save_image, (y, x,), radius, (0,255,0), 1)

    # Display the image, and also save it to a file
    cv2.imwrite("circles_only.jpg", save_image)
    cv2.imshow("result", save_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the key points, this will be used by task 3
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
    for image in gaussian_images:
        gx = cv2.filter2D(image.copy(), -1, dx)
        gy = cv2.filter2D(image.copy(), -1, dy)
        retval.append((gx, gy, ))
    return retval

"""
In task 3C, we need the weights for each pixel to multiply with the magnitude
of each point. This function gives us the weight of each point.
LRU cache is used to memoize the result once it is calculated
"""
@lru_cache
def get_weight(q:int, r:int, sigma:float)->float:
    wqr = np.exp( -(q**2 + r**2) / (9 * sigma**2 / 2 ))
    wqr = wqr / (9 * np.pi * sigma**2 / 2)
    return wqr

"""
For task 3C, this function multiplies the magnitude with the weight
for each point, and then uses this product to bin the angles
and then appends the angle to each key point

The magnitude and angle have already been calculated for all the points
and saved so that they needn't be calculated again and again
"""
def get_augmented_point_with_direction(\
        im:np.ndarray,\
        x:int,\
        y:int,\
        sigma:np.ndarray,\
        m_gx:np.ndarray,\
        m_gy:np.ndarray,\
        m_theta:np.ndarray,\
        m_magnitude:np.ndarray)->tuple:
    hist = [0.0] * 36
    for k in range(-3, 4):
        for j in range(-3, 4):
            q = round(3/2 * k * sigma)
            r = round(3/2 * j * sigma)
            xq = x + q
            yr = y + r
            mqr = m_magnitude[xq, yr]
            tqr = m_theta[xq, yr]
            wqr = get_weight(q, r, sigma)
            i = int(math.floor((tqr + np.pi) / (2 * np.pi / 36)))
            hist[i] += (wqr * mqr)
    thetahat = (2 * np.pi / 36) * (0.5 + np.argmax(np.array(hist))) - np.pi
    return (x, y, sigma, thetahat, )

"""
Display derivative of gaussian kernels
"""
def show_derivative_of_gaussian_images(images:list)->None:
    gxa = []
    gya = []
    for dog in images:
        gx = dog[0]
        gy = dog[1]
        gxa.append(gx)
        gya.append(gy)
    display_images_meshgrid(gxa, True, None)
    display_images_meshgrid(gya, True, None)

"""
Display derivative of gaussian kernels

The cross-ref we create in task_3() comes in very handy as we can just
lookup anything that we have already computed earlier in O(1) time.
"""
def plot_gx_gy(sigma_cross_ref:dict)->None:
    title_arr = []
    gx_arr = []
    gy_arr = []
    for i in sigma_cross_ref.keys():
        title_arr.append(f"sigma = {i:2.2f}")
        gx_arr.append(sigma_cross_ref[i]["gx"])
        gy_arr.append(sigma_cross_ref[i]["gy"])
    display_images_meshgrid(gx_arr, True, title_arr, "gx")
    display_images_meshgrid(gy_arr, True, title_arr, "gy")


def task_3()->tuple:
    # Get the grayscale image
    image = load_image_and_convert_to_grayscale()
    # Also load the original color image, which we will use
    # to draw everything at the end
    save_image = load_original_image()

    # Get the twelve smoothed images
    images, sigma_arr = get_twelve_smoothed_images(image)

    # Get the key points from task 2, this will be augmented by the direction
    key_points = task_2()

    # Get the derivative of gaussian images in x and y direction
    dog_images = get_derivative_of_gaussian(images, sigma_arr)
    magnitudes = []
    theta_array = []
    sigma_cross_ref = {}

    # Calculate the magnitude and angle for all points
    # We will select the points that matter to us later
    # This cross reference allows us to have a quick way to access
    # stuff that we've already calculated
    # task 4 will also use this cross ref, so we're calculating
    # some stuff here that is not strictly required in this task
    # as they will be required in task 4
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


    # Now plot the derivative of gaussian in x and y directions
    plot_gx_gy(sigma_cross_ref)
    plt.show()

    # For each key point, find the direction and augment that point
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
            
    # Now that we have found the direction theta for each point,
    # Draw a circle and a radius in the direction that we just found out
    # Also save the window
    save_image = load_original_image()
    for point in augmented_points:
        x, y, sigma, angle = point
        x = int(x)
        y = int(y)
        radius = math.floor(3 * sigma)
        cv2.circle(save_image, (y, x,), radius, (0,255,0), 1)
        x1 = int(round(x + radius * math.cos(angle)))
        y1 = int(round(y + radius * math.sin(angle)))
        cv2.line(save_image, (y, x, ), (y1, x1, ), (0,0,255,), 2)
    cv2.imwrite("circles_with_radius.jpg", save_image)
    cv2.imshow("result", save_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return augmented_points, sigma_cross_ref

"""The magnitude at each point will be multiplied by this weight in
task 4
"""
@lru_cache
def get_weight2(s:int, t:int, sigma:float)->float:
    return np.exp(-(s**2 + t**2) / (81 * sigma**2 /2)) / \
            (81 * np.pi * sigma**2 / 2)

def task_4()->None:
    # Get the key points augmented with the direction of each point,
    # as well as the sigma cross reference of all the stuff we've already
    # calculated at that sigma in task 2
    augmented_points, sigma_crf = task_3()

    # This array will hold all the SIFT descriptors for all the points
    all_descriptors = []

    # Now iterate all the points and find out the SIFT descriptor for each
    # point
    for point in augmented_points:
        try:
            x, y, sigma, theta_hat = point
            theta_arr = sigma_crf[sigma]["theta"]
            magnitude_arr = sigma_crf[sigma]["magnitude"]
            all_hist = []

            # Break the pixels around the key point into 16 4x4 grids
            # two nested loops to form 16 4x4 grids
            for i in range(-2, 2):
                for j in range(-2, 2):

                    # These will be used to calculate the coordinates inside
                    # each 4x4 grid
                    kk1 = [4 * i + x for x in range(4)]
                    kk2 = [4 * j + x for x in range(4)]

                    # This is the orientation histogram for each 4x4 subgrid
                    hist = [0.0] * 8 

                    # For each of the pixels in the 4x4 grid
                    for k1 in kk1:
                        for k2 in kk2:

                            s = int(round(9/16 * (k1 + 0.5) * sigma))
                            t = int(round(9/16 * (k2 + 0.5) * sigma))
                            wst = get_weight2(s, t, sigma)
                            xs = x + s
                            yt = y + t

                            # Find the magnitude
                            mst = magnitude_arr[xs, yt]

                            # Find the angle and subtract theta_hat from it
                            theta_st = theta_arr[xs, yt] - theta_hat

                            # Find the index in the histogram for this
                            # 4x4 subgrid from the angle
                            ind = int(math.floor(theta_st / (2 * np.pi / 8)))

                            # Update the weight in that index of the histogram
                            hist[ind] += (wst * mst)

                    # Now that we're done calculating the histogram for this
                    # 4x4 sub-grid, append it to the list of histograms for
                    # this point to form the descriptor
                    # At the end of all loops for this point, this will have
                    # 128 floats
                    all_hist.extend(hist)

            # append the descriptors for this point into the global list
            # of descriptors.
            # Each descriptor for each point has 128 floats
            all_hist = np.array(all_hist)
            all_hist = all_hist / np.sqrt(np.sum(np.square(all_hist)))
            all_hist[all_hist > 0.2] = 0.2
            all_descriptors.append(all_hist)
        except:
            pass
    for descriptor in all_descriptors[0:5]:
        print(len(descriptor), descriptor)

def main():
    task_1b()
    # Task 2 and Task 3 will automatically be called by task 4
    task_4()
    plt.show()

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="the image file name", type=str,
                        required=True)
    args = parser.parse_args()
    g_filename = args.file
    main()


