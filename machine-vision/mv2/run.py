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
PLAY_VIDEO = False
DRAW_CHECKERBOARD = False
F_ITERATIONS = 10

def get_checkerboard(gridsize):
    imfiles = glob.glob("Assignment_MV_02_calibration*.png")
    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in imfiles]
    or_images = [cv2.imread(fname) for fname in imfiles]
    corner_array_subpix = []
    for n, image in enumerate(images):
        ret, corners = cv2.findChessboardCorners(image, gridsize, 11)
        if ret:
            corners = cv2.cornerSubPix(image, corners, (11, 11), \
                    (-1, -1), STOP_CRITERIA)
            if DRAW_CHECKERBOARD:
                cv2.drawChessboardCorners(or_images[n], gridsize, corners, ret)
                cv2.imshow('image', or_images[n])
                cv2.waitKey(0)
            corner_array_subpix.append(corners)
        else:
            corner_array_subpix.append([])
    return images, or_images, corner_array_subpix

def get_calibration_matrix(gridsize, images, corner_array_subpix):
    object3d = np.zeros((1, gridsize[0] * gridsize[1], 3), np.float32)
    object3d[0,:,:2] = np.mgrid[0:gridsize[0], 0:gridsize[1]].T.reshape(-1, 2)
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
    return len(frames), frames, gray_frames



def get_K_matrix():
    images, or_images, corner_array_subpix = get_checkerboard(GRIDSIZE)
    ret, K, distortion, R, T = get_calibration_matrix(GRIDSIZE, images,\
            corner_array_subpix)
    return K

def get_homogenous(x):
    x = x.flatten()
    return np.array([x[0], x[1], 1])

# Takes two arrays and creates an array of tuples
def get_correspondence_array(X1, X2, K):
    ret = []
    directions = []
    for i, j in zip(X1, X2):
        # Convert to homogenous coordinates
        i = np.append(i.flatten().reshape(1,2), np.ones((1, 1))).reshape(3,1)
        j = np.append(j.flatten().reshape(1,2), np.ones((1, 1))).reshape(3,1)
        ret.append((i, j,))
        m = np.linalg.inv(K) @ i
        n = np.linalg.inv(K) @ j
        directions.append((m, n,))
    return ret, directions

def get_correspondences(frames, gray_frames):
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], 200, 0.3, 7) # shape = 109,1,2
    p0 = cv2.cornerSubPix(gray_frames[0], p0, (11, 11),\
            (-1, -1), STOP_CRITERIA)
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
            p1 = cv2.cornerSubPix(old_gray, p1, (11, 11), \
                    (-1, -1), STOP_CRITERIA)
            frame = frame.copy()
            for m, point in enumerate(p1[status.flatten() == 1]):
                cv2.circle(frame, (int(point[0,0]), int(point[0,1])), \
                        2, (255,0,0), 2)
            if PLAY_VIDEO:
                cv2.imshow('frame', frame)
                cv2.waitKey(VIDEO_DELAY)
            p = p1
            old_gray = gray
            history.append(p)

    x1 = np.zeros((0, 3,), dtype=np.float32)
    x2 = np.zeros((0, 3,), dtype=np.float32)
    for i, j in zip(original_points[status.flatten() == 1], \
                    p1[status.flatten() == 1]):
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
            cv2.line(frame, j, k, (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    return original_points[status.flatten() == 1], p1[status.flatten() == 1], \
            (x1, x2), history, status.flatten()

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

    # NOTE:
    # The problem description says that gi and s^2 must be calculated
    # only for the other points, but we do it for all the points
    # anyway because it makes it easier for the caller to work with the
    # original indices. The 8 points included here would all be inliers
    #
    # Must take the original points before multiplication with T1 and T2
    yy1 = p1_list.copy() #[chosen == False]
    yy2 = p2_list.copy() #[chosen == False]

    gi = np.zeros((0, 1))
    sigma2 = np.zeros((0, 1))
    
    for x1, x2 in zip(yy1, yy2):
        x1 = x1.reshape((3,1,))
        x2 = x2.reshape((3,1,))
        gi = np.append(gi, x2.T @ F @ x1)
        s2 = x1.T @ F @ CXX @ F.T @ x1 + x2.T @ F.T @ CXX @ F @ x2
        sigma2 = np.append(sigma2, s2.reshape((1,)))


    T = np.square(gi) / sigma2
    is_outlier = (T > 6.635)
    # To calculate the sum of inliers, only consider points
    # other than the 8 we used to calculate the F matrix
    inliers_sum = sum(T[chosen == False][T[chosen == False] <= 6.635])
    # While calculating number of outliers, only consider those points
    # other than the 8 we used to calculate the F matrix
    n_outliers = np.sum(is_outlier[chosen == False])

    return F, n_outliers, inliers_sum, is_outlier

def plot_tracks(frames, history, is_outlier_array, e1, e2, desc):
    e1 = tuple(e1.astype(int).tolist()[:2])
    e2 = tuple(e2.astype(int).tolist()[:2])
    for i in range(1, len(frames)):
        frame = frames[i].copy()
        cv2.circle(frame, e1, 2, (0, 255, 0), 5)
        cv2.circle(frame, e2, 2, (0, 0, 255), 5)
        for j in range(i):
            p1s = history[j]
            p2s = history[j+1]
            for x, y in zip(p1s[is_outlier_array], p2s[is_outlier_array]):
                cv2.line(frame, tuple(x.flatten().astype(int)), \
                        tuple(y.flatten().astype(int)), (100, 100, 255), 2)
            for x, y in zip(p1s[is_outlier_array == False], \
                            p2s[is_outlier_array == False]):
                cv2.line(frame, tuple(x.flatten().astype(int)), \
                        tuple(y.flatten().astype(int)), (255, 0, 0), 2)
        if PLAY_VIDEO:
            cv2.imshow(desc, frame)
            cv2.waitKey(VIDEO_DELAY)
    cv2.waitKey(0)

def get_best_fundamental_matrix(correspond):
    n_outliers = None
    F = None
    outliers_array = []
    least_outliers = len(correspond[0])
    max_inliers_sum = 0
    is_outlier_array = None
    for i in range(F_ITERATIONS):
        f, n_out, inliers_sum, outlier_arr = \
                get_fundamental_matrix(correspond[0], correspond[1])
        if n_out < least_outliers or \
                (n_out == least_outliers and max_inliers_sum < inliers_sum):
            least_outliers = n_out
            max_inliers_sum = inliers_sum
            F = f
            is_outlier_array = outlier_arr
    return F, n_outliers, outliers_array, is_outlier_array
            
def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]
    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]
    return e1,e2    

def get_essential_matrix(K, F):
    E = np.linalg.inv(K) @ F @ K
    U,S,V = np.linalg.svd(E)
    # Ensure S[0] and S[1] are the same
    S[0] = (S[0] + S[1]) / 2
    S[1] = S[0]
    # S[2] is turning out to be very small (e-15) but not zero, force zero
    S[2] = 0                
    # Ensure U and V have non negative determinant
    if np.linalg.det(U) < 0:
        U[:,2] *= -1
    if np.linalg.det(V) < 0:
        V[:,2] *= -1
    # Reconstruct fixed E
    E = U @ np.diag(S) @ V.T
    return E, U, S, V

def get_w_v():
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    return W, Z

# Get the two possible translation and two possible rotation matrices
# These can be combined into four combinations
def get_translation_rotation(U, S, V, beta=1):
    W, Z = get_w_v()
    T1 = beta * (U @ Z @ U.T)
    T2 = -1 * T1
    R1_T = U @ W @ V.T
    R2_T = U @ W.T @ V.T
    return T1, T2, R1_T, R2_T

def get_distance_from_speed(fps, n_frames, speed):
    t = n_frames / fps
    return speed * t * 5. / 18.

def main():
    K = get_K_matrix()
    n_frames, frames, gray_frames = get_frames_for_video()
    first_frame_points, last_frame_points, correspond, history, status = \
                    get_correspondences(frames, gray_frames)
    F, n_outliers, outliers_array, is_outlier_array = \
                    get_best_fundamental_matrix(correspond)

    # Get rid of all the points that disappeared somewhere in between
    for n, h in enumerate(history):
        history[n] = h[status == 1]

    e1, e2 = calculate_epipoles(F)
    e1 = np.divide(e1, e1[2])
    e2 = np.divide(e2, e2[2])
    plot_tracks(frames, history, is_outlier_array, e1, e2, \
            "Plot inliers and outliers tracks")

    X1 = history[0][is_outlier_array == False]
    X2 = history[-1][is_outlier_array == False]
    cor_points_x, cor_directions_m = get_correspondence_array(X1, X2, K)


    E, E_U, E_S, E_V = get_essential_matrix(K, F)
    T1, T2, R1_T, R2_T = get_translation_rotation(E_U, E_S, E_V,\
                                beta=get_distance_from_speed(30, n_frames, 50))
    t1 = T1[:,2].reshape(3,1)
    t2 = T2[:,2].reshape(3,1)

    print("T1")
    print(T1)
    print(t1)
    print("T2")
    print(T2)
    print(t2)
    print("R1")
    print(R1_T.T)
    print("R2")
    print(R2_T.T)

    print ("R = ", R1_T.shape)
    print ("T = ", T1.shape)
    print ("m = ", cor_directions_m[0][0].shape)
    print ("m = ", cor_points_x[0][0].shape)

    def solve(m, md, R, t):
        x1 = t.T @ m
        x2 = t.T @ R @ md
        RHM = np.append(x1, x2, axis=0)

        x1 = np.matmul(m.T, m).flatten()[0]
        x2 = -1 * np.matmul(m.T, np.matmul(R, md)).flatten()[0]
        x3 = np.matmul(m.T, np.matmul(R, md)).flatten()[0]
        x4 = -1 * np.matmul(md.T, md).flatten()[0]
        LHM = np.array([[x1, x2], [x3, x4]])

        uv = np.linalg.inv(LHM) @ RHM

        return tuple(uv.flatten().tolist())


    m = cor_directions_m[0][0]
    md = cor_directions_m[0][1]

    u, v = solve(m, md, R1_T.T, t1)
    if not (u >= 0 and v >= 0):
        u, v = solve(m, md, R1_T.T, t2)
    if not (u >= 0 and v >= 0):
        u, v = solve(m, md, R2_T.T, t1)
    if not (u >= 0 and v >= 0):
        u, v = solve(m, md, R2_T.T, t2)

    l, mu = u, v




if "__main__" == __name__:
    main()
