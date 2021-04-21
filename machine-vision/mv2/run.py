#!/usr/bin/env python3

import os
import cv2
import math
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
GRIDSIZE = (5, 7, )
VIDEO_DELAY = 1

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

def get_frames_for_video():
    frames = list()
    gray_frames = list()
    cap = cv2.VideoCapture("./Assignment_MV_02_video.mp4")
    while cap.grab():
        ret, frame = cap.retrieve()
        if False != ret:
            gray = frame.copy()
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            gray_frames.append(gray)
    cap.release()
    cv2.destroyAllWindows()
    return frames, gray_frames



def get_K_matrix():
    images, or_images, corner_array_subpix = get_checkerboard(GRIDSIZE)
    ret, K, distortion, R, T = get_calibration_matrix(GRIDSIZE, images, corner_array_subpix)
    return K

def get_homogenous(x):
    x = x.flatten()
    return np.array([x[0], x[1], 1])

def get_correspondences(frames, gray_frames):
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], 200, 0.3, 7) # shape = 109,1,2
    p0 = cv2.cornerSubPix(gray_frames[0], p0, (11, 11), (-1, -1), STOP_CRITERIA)
    original_points = p0.copy()
    old_gray = gray_frames[0]
    frames = frames[1:]
    gray_frames = gray_frames[1:]
    history = []
    history.append(p0)

    if len(p0) > 0:
        p = p0
        for n, (frame, gray,) in enumerate(zip(frames, gray_frames)):
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p, None)
            p1 = cv2.cornerSubPix(old_gray, p1, (11, 11), (-1, -1), STOP_CRITERIA)
            frame = frame.copy()
            for m, point in enumerate(p1[status.flatten() == 1]):
                cv2.circle(frame, (int(point[0,0]), int(point[0,1])), 2, 0xff0000, 2)
            cv2.imshow('frame', frame)
            cv2.waitKey(VIDEO_DELAY)
            p = p1
            old_gray = gray
            history.append(p)

    x1 = np.zeros((0, 3,), dtype=np.float32)
    x2 = np.zeros((0, 3,), dtype=np.float32)
    for i, j in zip(original_points[status.flatten() == 1], p1[status.flatten() == 1]):
        i = get_homogenous(i)
        j = get_homogenous(j)
        x1 = np.append(x1, i.reshape(1, 3), axis=0)
        x2 = np.append(x2, j.reshape(1, 3), axis=0)

    frame = frames[0].copy()
    for i in range(len(history) - 1):
        h0 = history[i][status.flatten() == 1]
        h1 = history[i+1][status.flatten() == 1]
        for j, k in zip(h0, h1):
            j = tuple(j.flatten().astype(int))
            k = tuple(k.flatten().astype(int))
            cv2.line(frame, j, k, 0x0000ff, 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    return original_points[status.flatten() == 1], p1[status.flatten() == 1], (x1, x2), history

CXX = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

def get_fundamental_matrix(p1_list, p2_list):
    def get_T(m, s):
        T = np.array([
            [   1/s[0],    0,          -m[0]/s[0],  ],
            [   0,         1/s[1],     -m[1]/s[1],  ],
            [   0,         0,          1,           ],
            ])
        return T
    mu1 = np.mean(p1_list, axis=0)
    mu2 = np.mean(p2_list, axis=0)
    sigma1 = np.std(p1_list, axis=0)
    sigma2 = np.std(p2_list, axis=0)
    T1 = get_T(mu1, sigma1)
    T2 = get_T(mu2, sigma2)
    y1 = np.matmul(T1, p1_list.T).T
    y2 = np.matmul(T2, p2_list.T).T
    chosen = np.array([False] * y1.shape[0], dtype=bool)
    chosen[np.random.choice(y1.shape[0], 8, replace=False)] = True
    yy1 = y1[chosen]
    yy2 = y2[chosen]

    A = np.zeros((0,9), dtype=np.float32)
    for x1, x2 in zip(yy1, yy2):
        A = np.append(A, [np.kron(x1.T, x2.T)], axis=0)
    U,S,V = np.linalg.svd(A)
    F = V[8,:].reshape(3,3).T
    # Enforce singularity
    U,S,V = np.linalg.svd(F)
    F = np.matmul(U, np.matmul(np.diag([S[0],S[1],0]), V))
    # Verify it is indeed singular
    """
    if np.linalg.det(F) < 1.0e-10:
        print("F is singular")
    else:
        print(f"F is not singular {np.linalg.det(F)}")
    """
    assert(np.linalg.det(F) < 1.0e-10)
    F = T2.T @ F @ T1
    """
    if np.linalg.det(F) < 1.0e-10:
        print("F is singular")
    else:
        print(f"F is not singular {np.linalg.det(F)}")
    """
    assert(np.linalg.det(F) < 1.0e-10)

    # Must take the original points before multiplication with T1 and T2
    yy1 = p1_list[chosen == False]
    yy2 = p2_list[chosen == False]
    gi = np.zeros((0, 1))
    
    kkkk = 0
    for x1, x2 in zip(yy1, yy2):
        kkkk += 1
        x1 = x1.reshape((3,1,))
        x2 = x2.reshape((3,1,))
        gi = np.append(gi, x2.T @ F @ x1)

    sigma2 = np.zeros((0, 1))
    for x1, x2 in zip(yy1, yy2):
        x1 = x1.reshape((3,1,))
        x2 = x2.reshape((3,1,))
        s2 = x1.T @ F @ CXX @ F.T @ x1 + x2.T @ F.T @ CXX @ F @ x2
        sigma2 = np.append(sigma2, s2.reshape((1,)))
    T = np.square(gi) / sigma2
    is_outlier = (T > 6.635)
    inliers_sum = sum(T[T <= 6.635])
    n_outliers = np.sum(is_outlier)

    return F, n_outliers, inliers_sum, is_outlier



#K = get_K_matrix()
frames, gray_frames = get_frames_for_video()
first_frame_points, last_frame_points, correspond, history = get_correspondences(frames, gray_frames)
outliers = []
F_matrices = []
outliers_array = []
for i in range(10000):
    F, n_out, inliers_sum, outlier_arr = get_fundamental_matrix(correspond[0], correspond[1])
    outliers.append(n_out)
    F_matrices.append(F)
    outliers_array.append(outlier_arr)
index = np.argmin(np.array(outliers))
print(outliers[index])
print(F_matrices[index])
