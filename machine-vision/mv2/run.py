#!/usr/bin/env python3 

import os
import cv2
import math
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import sys


STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
GRIDSIZE = (5, 7, )
VIDEO_DELAY = int(1 / 30 * 1000)
PLAY_VIDEO = True
DRAW_CHECKERBOARD = True
F_ITERATIONS = 10_000
DEBUG = False
PLOT_X_LAMBDA = True 
PLOT_X_MU = True
RANDOM_SEED = 22


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def wait_for_key_press():
    print("\nPress enter to continue ... ")
    cv2.waitKey(0)
    print("Continuing ... OK\n")

def plot_show():
    print("\nClose the plot to continue ...")
    plt.show()
    print("Continuing ... OK\n")

# TASK 1 Part A
# Read all calibration images, find the corners, and display
# the checkerboard
def get_checkerboard(gridsize):
    # Look for the calibration images in teh current directory
    imfiles = glob.glob("Assignment_MV_02_calibration*.png")

    # Read the files as images
    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in imfiles]

    # Read the files again for display purposes later
    or_images = [cv2.imread(fname) for fname in imfiles]
    corner_array_subpix = []

    # For each image, find the corners
    for n, image in enumerate(images):
        ret, corners = cv2.findChessboardCorners(image, gridsize, 11)
        if ret:
            corners = cv2.cornerSubPix(image, corners, (11, 11), \
                    (-1, -1), STOP_CRITERIA)
            if DRAW_CHECKERBOARD:
                cv2.drawChessboardCorners(or_images[n], gridsize, corners, ret)
                cv2.imshow(f"checkerboard {n}", or_images[n])
            corner_array_subpix.append(corners)
        else:
            corner_array_subpix.append([])
    wait_for_key_press()
    return images, or_images, corner_array_subpix

# Task 1 Part B
# Get the calibration matrix K using opencv
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
    print(f"\nImage Aspect Ratio is {images[0].shape[1] / images[0].shape[0]} : 1")
    print(f"\nFocal Length Aspect Ratio is {matrix[0,0]/matrix[1,1]} : 1")
    print(f"\nFocal lengths = {matrix[0, 0]}, {matrix[1, 1]}")
    print(f"\nPrincipal point = ({matrix[0, 2]},{matrix[1, 2]})")
    print(f"\nCalibration Matrix = \n{matrix}\n")
    return ret, matrix, distortion, rotation, translation

# Task 1 Part C
# Open the video and get the video
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

# Part 1 Task C and D, and Part 2 Task A
def get_correspondences(frames, gray_frames):
    # Get good features to track
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], 200, 0.3, 7) # shape = 109,1,2

    # Refine the points to sub-pixel accuracy
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

            # Track the features using OpenCV's KLT implementation
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p, None)

            # Refine points to sub-pixel accuracy
            p1 = cv2.cornerSubPix(old_gray, p1, (11, 11), \
                    (-1, -1), STOP_CRITERIA)
            frame = frame.copy()

            # Visualize the tracking
            for m, point in enumerate(p1[status.flatten() == 1]):
                cv2.circle(frame, (int(point[0,0]), int(point[0,1])), \
                        2, (255,0,0), 2)
            if PLAY_VIDEO:
                cv2.imshow('Playing tracks', frame)
                cv2.waitKey(VIDEO_DELAY)
            p = p1
            old_gray = gray
            history.append(p)

    # Task 2 Part A
    x1 = np.zeros((0, 3,), dtype=np.float32)
    x2 = np.zeros((0, 3,), dtype=np.float32)
    for i, j in zip(original_points[status.flatten() == 1], \
                    p1[status.flatten() == 1]):
        # Get Normalized homogenous vectors for all tracks
        i = get_homogenous(i)
        j = get_homogenous(j)
        x1 = np.append(x1, i.reshape(1, 3), axis=0)
        x2 = np.append(x2, j.reshape(1, 3), axis=0)

    # Plot all the tracks
    # Every time we called calcOpticalFlowPyrLK, we got two arrays
    # 1. An array to indicate where the new point is
    # 2. An array to indicate whether the point is still valid or not
    #
    # Both these arrays are of the same dimensions as the original array
    # 
    # What we could have done was to weed out the invalid points and create
    # a shorter array the next time we called calcOpticalFlowPyrLK, but
    # this would have complicated things
    #
    # Instead we pass this the same array (which now has some invalid points)
    # An invalid point will still be invalid on output from the function
    # and the function does not behave improperly
    #
    # The in the last step, we only choose those points which are valid in the
    # last frame
    #
    # Each time the function returns a set of points, that is stored in a
    # history array, all elements of history array are of same length for simplicity
    frame = frames[0].copy()
    for i in range(len(history) - 1):
        h0 = history[i][status.flatten() == 1]
        h1 = history[i+1][status.flatten() == 1]
        for j, k in zip(h0, h1):
            j = tuple(j.flatten().astype(int))
            k = tuple(k.flatten().astype(int))
            cv2.line(frame, j, k, (0,0,255), 2)
    cv2.imshow('Showing tracks', frame)
    return original_points[status.flatten() == 1], p1[status.flatten() == 1], \
            (x1, x2), history, status.flatten()


# Task 2 Part B, C, D, E, F
CXX = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
def get_fundamental_matrix(p1_list, p2_list):
    # p1_list and p2_list are numpy arrays of dimensions
    # N_points x 3
    # One denotes the first frame, the other the last frame

    # Task 2 Part B

    FORCE_POSITIVE_DETERMINANT = False

    # Setting it to false because it prints 100000 lines otherwise
    CHECK_SINGULARITY_ON_EACH_ITERATION = False


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

    # Task 2 Part C

    # Choose 8 random points without replacement
    chosen = np.array([False] * y1.shape[0], dtype=bool)
    chosen[np.random.choice(y1.shape[0], 8, replace=False)] = True
    yy1 = y1[chosen]
    yy2 = y2[chosen]

    # For the chosen points, calculate the A matrix
    A = np.zeros((0,9), dtype=np.float32)
    for x1, x2 in zip(yy1, yy2):
        A = np.append(A, [np.kron(x1.T, x2.T)], axis=0)

    # Task 2 Part D

    U,S,V = np.linalg.svd(A)

    # Calculate F
    F = V[8,:].reshape(3,3).T

    # Enforce singularity
    U,S,V = np.linalg.svd(F)

    # Force the last diagnoal element to be 0, this is F-hat
    F = np.matmul(U, np.matmul(np.diag([S[0],S[1],0]), V))

    # Verify it is indeed singular
    if CHECK_SINGULARITY_ON_EACH_ITERATION:
        if np.abs(np.linalg.det(F)) < 1.0e-10:
            print("F is singular")
        else:
            print(f"F is not singular {np.linalg.det(F)}")

    # Get the actual Fundamental matrix from the normalized F-hat
    F = T2.T @ F @ T1

    # Assert singularity
    assert(np.abs(np.linalg.det(F)) < 1.0e-10)


    if CHECK_SINGULARITY_ON_EACH_ITERATION:
        if np.linalg.det(F) < 1.0e-10:
            print("F is singular")
        else:
            print(f"F is not singular {np.linalg.det(F)}")

    # Assert singularity
    assert(np.abs(np.linalg.det(F)) < 1.0e-10)

    Task 2 Part E
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
        s2 = x2.T @ F @ CXX @ F.T @ x2 + x1.T @ F.T @ CXX @ F @ x1
        sigma2 = np.append(sigma2, s2.reshape((1,)))


    # Task 2 Part F

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

def unskew(m):
    x1 = m[2][1]
    x2 = m[0][2]
    x3 = m[1][0]
    return np.array([[x1], [x2], [x3]])

def get_best_fundamental_matrix(correspond):
    n_outliers = None
    F = None
    outliers_array = []
    least_outliers = len(correspond[0])
    max_inliers_sum = 0
    is_outlier_array = None
    print(f"\n\nIterating {F_ITERATIONS} times. Please wait for next prompt")
    for i in range(F_ITERATIONS):
        if i != 0 and 0 == (i % (F_ITERATIONS / 10)):
            print(f"Completed {i} iterations, wait for next prompt")
        f, n_out, inliers_sum, outlier_arr = \
                get_fundamental_matrix(correspond[0], correspond[1])
        if n_out < least_outliers or \
                (n_out == least_outliers and max_inliers_sum < inliers_sum):
            least_outliers = n_out
            max_inliers_sum = inliers_sum
            F = f
            is_outlier_array = outlier_arr
    print("OK\n")
    if np.linalg.det(F) < 1.0e-10:
        print("F is singular")
    else:
        print(f"F is not singular {np.linalg.det(F)}")

    print("Fundamental Matrix =\n", F, "\n")
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
    S[0] = S[1] = ((S[0] + S[1]) / 2)

    # S[2] is turning out to be very small (e-15) but not zero, force zero
    S[2] = 0.

    # Ensure U and V have non negative determinant
    if np.linalg.det(U) < 0:
        U[:,2] *= -1
    if np.linalg.det(V) < 0:
        V[2,:] *= -1

    # Reconstruct fixed E
    E = U @ np.diag(S) @ V

    # Task: Make sure that S[0] and S[1] are the same
    assert(np.abs(S[0] - S[1]) < 1.0e-10)

    print("Essential matrix =\n", E)
    return E, U, S, V

def get_w_v():
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    return W, Z

# Get the two possible translation and two possible rotation matrices
# These can be combined into four combinations
def get_translation_rotation(U, S, V, beta=1):
    matrices = []
    W, Z = get_w_v()
    RT_T1 = unskew(beta * (U @ Z @ U.T))
    RT_T2 = -1 * RT_T1
    R1_T = U @ W @ V.T
    R2_T = U @ W.T @ V.T

    assert(np.linalg.det(R1_T.T) > 0 and np.linalg.det(R2_T.T) > 0)

    def append_matrices(matrices, R_T, R_T_T):
        T = np.linalg.inv(R_T) @ R_T_T
        matrices.append((R_T.T, T, ))

    append_matrices(matrices, R1_T, RT_T1)
    append_matrices(matrices, R2_T, RT_T1)
    append_matrices(matrices, R1_T, RT_T2)
    append_matrices(matrices, R2_T, RT_T2)

    return matrices


def get_distance_from_speed(fps, n_frames, speed):
    t = n_frames / fps
    return speed * t * 5. / 18.

def print_validation_matrix(E, beta):
    if __debug__ and DEBUG:
        R1, R2, T = cv2.decomposeEssentialMat(E)
        print('-' * 80)
        print("From CV2")
        print('-' * 80)
        print(f"cv2 T1        = \n{T * beta}")
        print(f"cv2 R1        = \n{R1.T}")
        print(f"cv2 R2        = \n{R2.T}")
        print('=' * 80)

def solve(m, md, R, t):
    """
    r1 = m.T @ m
    r1 = np.append(r1, -1 * (m.T @ R @ md), axis=1)
    r2 = m.T @ R @ md
    r2 = np.append(r2, -1 * (md.T @ md), axis=1)
    LHS1 = np.append(r1, r2, axis=0)
    """
    a = m.T @ m
    a = a.flatten().tolist()[0]
    b = -1 * (m.T @ R @ md)
    b = b.flatten().tolist()[0]
    c = m.T @ R @ md
    c = c.flatten().tolist()[0]
    d = -1 * (md.T @ md)
    d = d.flatten().tolist()[0]
    LHS = np.array([[a, b], [c, d]])

    """
    for i in range(4):
        print("asserting", i)
        assert(LHS1.flatten().tolist()[i] == LHS.flatten().tolist()[i])
    """

    if t.shape == (3, 3):
        # convert 3x3 skew symmetric matrix to a 3x1 matrix
        t = unskew(t)
    if __debug__ and DEBUG:
        print(t)

    r1 = t.T @ m
    r2 = t.T @ R @ md
    RHS = np.append(r1, r2, axis=0)

    lambda_mu = np.linalg.inv(LHS) @ RHS

    return tuple(lambda_mu.flatten().tolist())

def get_best_r_t(r_t_list, cor_directions_m):
    # For each combination R, T, calculate the lambda, mu
    # for each set of correspondences
    # Find the R, T where the biggest number of inlier
    # points are there
    # Inliers are those where lambda >= 0 and mu >= 0
    best_positive_lambda_mu_count = 0
    best_R = None
    best_T = None

    for (R, T) in r_t_list:
        positive_lambda_mu_count = 0
        for (m, md) in cor_directions_m:
            lmbda , mu = solve(m, md, R, T)
            if lmbda >= 0 and mu >= 0:
                positive_lambda_mu_count += 1
        if positive_lambda_mu_count > best_positive_lambda_mu_count:
            best_positive_lambda_mu_count = positive_lambda_mu_count
            best_R = R
            best_T = T

    print("\nChosen Rotation Matrix =\n", best_R)
    print("Determinant of R = ", np.linalg.det(R), "\n")
    print("\nChosen Translation Matrix =\n", best_T)
    return best_R, best_T

def get_inlier_correspondences(R, T, cor_points_x, cor_directions_m):
    # Find all outliers, that is ones which are behind
    # the camera, where either lambda or mu are negative
    out_cor_points = []
    out_cor_directions = []
    for i, (m, md) in enumerate(cor_directions_m):
        lmbda, mu = solve(m, md, R, T)
        if lmbda >= 0 and mu >= 0:
            out_cor_points.append(cor_points_x[i])
            out_cor_directions.append(cor_directions_m[i])
    return out_cor_points, out_cor_directions

def verify_directions_converge(cor_directions_m):
    # Print out all the intersections of the directions
    # They should all intersect at the camera centre
    # or very close to it.
    # if they doin't then there is a problem
    # Vsually verify
    if __debug__ and DEBUG:
        for i in range(len(cor_directions_m) - 1):
            for j in range(i, len(cor_directions_m)):
                m1 = cor_directions_m[i][1]
                m2 = cor_directions_m[j][1]
                m1 = np.reshape(m1, -1)
                m2 = np.reshape(m2, -1)
                print(np.cross(m1, m2))

def get_3d_points(R, T, cor_directions_m):
    mean_three_d_points = list()
    x_lambda_points = list()
    x_mu_points = list()
    for i, (m, md) in enumerate(cor_directions_m):
        lmbda, mu = solve(m, md, R, T)
        x_lmbda = lmbda * m
        x_mu = T + (mu * (R @ md))
        if __debug__ and DEBUG:
            xx, yy = normalize(x_lmbda.flatten()),\
                            normalize(x_mu.flatten())
            print(i, xx, yy, np.sum(np.square(xx - yy)))
        x_average = (x_lmbda + x_mu) / 2
        x_lambda_points.append(x_lmbda)
        x_mu_points.append(x_mu)
        mean_three_d_points.append(x_average)
    return mean_three_d_points, x_lambda_points, x_mu_points

def get_camera_centres(R, T):
    c1 = np.reshape(np.array([0, 0, 0]), (3, 1, ))
    c2 = T + c1
    return c1, c2

class AnnotateThreeDimension(Annotation):
    def __init__(self, s, coords, *args, **kwargs):
        Annotation.__init__(self, s, *args, **kwargs)
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]

    def draw(self, renderer):
        x, y, z = proj_transform(self.x, self.y, self.z, renderer.M)
        self.xy=(x, y,)
        Annotation.draw(self, renderer)

def create_3d_plot(c1, c2, three_d_points, lmbda_pt, mu_pt):
    def plot_point(ax, p, clr='red', txt=None, markr=None, msz=-1):
        x, y, z = tuple(p.tolist())
        if markr and msz != -1:
            ax.scatter3D(x, y, z, color=clr, marker=markr, s=msz) 
        else:
            ax.scatter3D(x, y, z, color=clr)
        # For some reason adding text labels doesn't quite work as expected
        # Hence we'll just force this to be NONE for the moment
        txt = None
        if None != txt:
            ax.text(x, y, z, txt, zorder=1, size=20, color='k')
            AnnotateThreeDimension(txt, p.tolist(), 
                    ax, fontsize=10, xytext=(-3,-3))
                    #textcoords='offset points', ha='right',va='bottom')

    def plot_line(ax, p1, p2, clr='red'):
        p1 = p1.flatten().tolist()
        p2 = p2.flatten().tolist()
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]
        ax.plot(x, y, z, color=clr)


    fig = plt.figure()
    ax = fig.gca(projection ="3d")
    plot_point(ax, c1, clr='blue', txt='c1', markr='P', msz=198)
    plot_point(ax, c2, clr='blue', txt='c2', markr='P', msz=198)
    plot_line(ax, c1, c2, clr='tan')
    for p in three_d_points:
        plot_point(ax, p, markr='o')
    if PLOT_X_LAMBDA:
        for p in lmbda_pt:
            plot_point(ax, p, clr='green', markr='.')
    if PLOT_X_MU:
        for p in mu_pt:
            plot_point(ax, p, clr='cyan', markr='.')
    if PLOT_X_LAMBDA:
        for p1, p2 in zip(lmbda_pt, three_d_points):
            plot_line(ax, p1, p2, clr='black')
    if PLOT_X_MU:
        for p1, p2 in zip(mu_pt, three_d_points):
            plot_line(ax, p1, p2, clr='black')
    plt.title("3 D plot of world points, X[lambda], X[mu] and their mean")

def normalize(x):
    sh = x.shape
    x = x / np.reshape(x, -1)[-1]
    return np.reshape(x, sh)

def get_reprojected_points(three_d_points, K, R, T):
    def get_reprojected(p):
        x = normalize(K @ p)
        xd = normalize(K @ R.T @ (p - T))
        return x, xd

    reprojected = list()
    for p in three_d_points:
        x, xd = get_reprojected(p)
        reprojected.append((x, xd,))
        #print(np.reshape(p, -1), " ---> ", np.reshape(x,-1), np.reshape(xd,-1))
    return reprojected

def plot_reprojected(cor_x_pts, reprojected_x_pts):
    def get_xy(x):
        x = normalize(np.reshape(x, -1))
        return x[:2]

    def plot_rp(ax, orig, rep, clr1, clr2):
        ax.scatter(orig[0], orig[1], color=clr1)
        ax.scatter(rep[0], rep[1], color=clr2)
        x_values = [orig[0], rep[0]]
        y_values = [orig[1], rep[1]]
        ax.plot(x_values, y_values, color='black')

    # Original points are in blue, reprojected in red
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("X [avg of mu and lambda] plotted onto first and last frame")
    for orig, rep in zip(cor_x_pts, reprojected_x_pts):
        x, x1 = orig[0], rep[0]
        x = get_xy(x)
        x1 = get_xy(x1)
        plot_rp(ax[0], x, x1, 'blue', 'red')
        x, x1 = orig[1], rep[1]
        x = get_xy(x)
        x1 = get_xy(x1)
        plot_rp(ax[1], x, x1, 'blue', 'red')

def plot_reprojected_on_img(\
        first_frame,\
        last_frame,\
        cor_x_pts,\
        reprojected_pts):
    # Original points are in blue, reprojected in red

    # This is a different plot, instead of plotting
    # (X[mu] + X[lamgda])/2 into both the first and last frames
    # here we reproject X[lambda] onto the first frame
    # and reproject X[mu] onto the last frame
    # and plot both in the image and on a graph
    #
    # We see that the reprojections completely align in this
    # plot.

    first_frame = first_frame.copy()
    last_frame = last_frame.copy()
    def get_xy(x):
        x = normalize(np.reshape(x, -1))
        return x[:2]

    def plot_rp(fr, x, y):
        x = tuple(x.flatten().astype(int).tolist())
        y = tuple(y.flatten().astype(int).tolist())
        cv2.line(fr, x, y, color=(0,255,0), thickness=2)
        cv2.circle(fr, x, radius=4, color=(255,0,0), thickness=2)
        cv2.circle(fr, y, radius=4, color=(0,0,255), thickness=2)

    for orig, rep in zip(cor_x_pts, reprojected_pts):
        x, x1 = orig[0], rep[0]
        x = get_xy(x)
        x1 = get_xy(x1)
        plot_rp(first_frame, x, x1)

    print("\n\nShowing First frame ...")
    cv2.imshow("Reprojection of average(X[mu], X[lambda]) on first frame",\
            first_frame)
    print("OK")

    for orig, rep in zip(cor_x_pts, reprojected_pts):
        x, x1 = orig[1], rep[1]
        x = get_xy(x)
        x1 = get_xy(x1)
        plot_rp(last_frame, x, x1)

    print("\n\nShowing Last frame ...")
    cv2.imshow("Reprojection of average(X[mu], X[lambda]) on last frame",\
            last_frame)
    print("OK")
    pass

def plot_reprojected2(cor_x_pts, reprojected_x_pts):
    def get_xy(x):
        x = normalize(np.reshape(x, -1))
        return x[:2]

    def plot_rp(ax, orig, rep, clr1, clr2):
        ax.scatter(orig[0], orig[1], color=clr1)
        ax.scatter(rep[0], rep[1], color=clr2)
        x_values = [orig[0], rep[0]]
        y_values = [orig[1], rep[1]]
        ax.plot(x_values, y_values, color='black')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for orig, rep in zip(cor_x_pts, reprojected_x_pts):
        x = get_xy(orig)
        x1 = get_xy(rep)
        plot_rp(ax, x, x1, 'red', 'blue')

def plot_xlambda_xmu_reprojected_separately(\
        first_frame, last_frame, cor_points_x, x_lmbda_3d, x_mu_3d, K, R, T):

    def get_xy(x):
        x = normalize(np.reshape(x, -1))
        return x[:2]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("X[lambda] re-projected onto first frame and x[mu] to last")

    def plot_rp(ax, orig, rep, clr1, clr2):
        ax.scatter(orig[0], orig[1], color=clr1)
        ax.scatter(rep[0], rep[1], color=clr2)
        x_values = [orig[0], rep[0]]
        y_values = [orig[1], rep[1]]
        ax.plot(x_values, y_values, color='black')

    def plot_rp_img(fr, x, y):
        x = tuple(x.flatten().astype(int).tolist())
        y = tuple(y.flatten().astype(int).tolist())
        cv2.line(fr, x, y, color=(0,255,0), thickness=2)
        cv2.circle(fr, x, radius=4, color=(255,0,0), thickness=2)
        cv2.circle(fr, y, radius=8, color=(0,0,255), thickness=2)

    def plot_points(ax, frame, o, r, titstr):
        plot_rp(ax, o, r, "blue", "red")
        plot_rp_img(frame, o, r)
        pass


    first_frame = first_frame.copy()
    last_frame = last_frame.copy()
    reprojected_lambda = get_reprojected_points(x_lmbda_3d, K, R, T)
    reprojected_mu = get_reprojected_points(x_mu_3d, K, R, T)

    for orig, rep in zip(cor_points_x, reprojected_lambda):
        if __debug__ and DEBUG:
            print(orig[0].flatten())
            print(rep[0].flatten())
        o = get_xy(orig[0])
        r = get_xy(rep[0])
        plot_points(ax[0], first_frame, o, r, "x[lambda] on first frame")

    for orig, rep in zip(cor_points_x, reprojected_mu):
        o = get_xy(orig[1])
        r = get_xy(rep[1])
        plot_points(ax[1], last_frame, o, r, "x[mu] on last frame")

    cv2.imshow("X[lambda] re-projected onto first frame", first_frame)
    cv2.imshow("X[mu] re-projeced onto last frame", last_frame)

def main():
    # Task 1
    K = get_K_matrix()
    n_frames, frames, gray_frames = get_frames_for_video()

    # Get correspondence points
    first_frame_points, last_frame_points, correspond, history, status = \
                    get_correspondences(frames, gray_frames)

    # Task 2
    F, n_outliers, outliers_array, is_outlier_array = \
                    get_best_fundamental_matrix(correspond)

    # Get rid of all the points that disappeared somewhere in between
    for n, h in enumerate(history):
        history[n] = h[status == 1]

    # e1 and e2 are the epipoles
    e1, e2 = calculate_epipoles(F)
    e1 = np.divide(e1, e1[2])
    e2 = np.divide(e2, e2[2])

    # Plot the tracks in the image
    plot_tracks(frames, history, is_outlier_array, e1, e2, \
            "Plot inliers and outliers tracks")

    X1 = history[0][is_outlier_array == False]
    X2 = history[-1][is_outlier_array == False]
    cor_points_x, cor_directions_m = get_correspondence_array(X1, X2, K)


    # Task 3
    E, E_U, E_S, E_V = get_essential_matrix(K, F)
    r_t_matrices = get_translation_rotation(E_U, E_S, E_V,\
                                beta=get_distance_from_speed(30, n_frames, 50))
    print_validation_matrix(E, beta=get_distance_from_speed(30, n_frames, 50))


    # Get the R and T matrices that fit the best
    R, T = get_best_r_t(r_t_matrices, cor_directions_m)



    # Discard outliers (points behind either camera)
    cor_points_x, cor_directions_m = get_inlier_correspondences(\
            R, T, cor_points_x, cor_directions_m)
    verify_directions_converge(cor_directions_m)


    c1, c2 = get_camera_centres(R, T)
    mean_three_d_points, x_lmbda_3d, x_mu_3d = get_3d_points(\
                                                R, T, cor_directions_m)
    create_3d_plot(c1, c2, mean_three_d_points, x_lmbda_3d, x_mu_3d)
    
    # Get reprojected points
    mean_reprojected = get_reprojected_points(mean_three_d_points, K, R, T)

    plot_reprojected(cor_points_x, mean_reprojected)
    plot_reprojected_on_img(frames[0], frames[-1], \
                                cor_points_x, mean_reprojected)

    # Another experiment
    #
    # - Reproject X[lambda] on the first frame only
    # - Reproject X[mu] to the last frame only
    # and plot
    # Results - we see perfect reprojection
    plot_xlambda_xmu_reprojected_separately(\
            frames[0], frames[-1], cor_points_x, x_lmbda_3d, x_mu_3d, K, R, T)

    wait_for_key_press()
    plot_show()


if "__main__" == __name__:
    main()
