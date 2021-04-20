#!/usr/bin/env python3

import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
GRIDSIZE = (5, 7, )

def get_checkerboard(gridsize):
    image_files = glob.glob("Assignment_MV_02_calibration*.png")
    images = [cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY) for fname in image_files]
    or_images = [cv2.imread(fname) for fname in image_files]
    corner_array_subpix = []
    for n, image in enumerate(images):
        ret, corners = cv2.findChessboardCorners(image, gridsize, 11)
        if ret:
            corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), STOP_CRITERIA)
            #for c in corners:
            #    cv2.circle(or_images[n], (int(c[0][0]), int(c[0][1])), 3, 0xff0000, 5)
            cv2.drawChessboardCorners(or_images[n], gridsize, corners, ret)
            cv2.imshow('image', or_images[n])
            cv2.waitKey(0)
            corner_array_subpix.append(corners)
        else:
            corner_array_subpix.append([])
    return images, or_images, corner_array_subpix

def get_calibration_matrix(gridsize, images, corner_array_subpix):
    object3d = np.zeros((1, gridsize[0] * gridsize[1], 3), np.float32)
    object3d[0, :, :2] = np.mgrid[0:gridsize[0], 0:gridsize[1]].T.reshape(-1, 2)
    worldpoints = []
    camerapoints = []
    for corners in corner_array_subpix:
        if len(corners) > 0:
            worldpoints.append(object3d)
            camerapoints.append(corners)
    ret, matrix, distortion, rotation, translation = cv2.calibrateCamera(
        worldpoints, camerapoints, images[0].shape, None, None)
    print(f"Focal lengths are {matrix[0, 0]}, {matrix[1, 1]}")
    print(f"Principal point is ({matrix[0, 2]},{matrix[1, 2]})")
    print(f"Calibration matrix is: \n{matrix}\n")
    return ret, matrix, distortion, rotation, translation



def get_K_matrix():
    images, or_images, corner_array_subpix = get_checkerboard(GRIDSIZE)
    ret, K, distortion, R, T = get_calibration_matrix(GRIDSIZE, images, corner_array_subpix)
    return K

K = get_K_matrix()
