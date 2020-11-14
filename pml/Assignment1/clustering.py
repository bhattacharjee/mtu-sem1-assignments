#!/usr/bin/env python3

import math
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import random
import pickle

g_apply_scaling = False
g_apply_normalization = False

random.seed(12345)
np.random.seed(12345)

def calculate_distances(allvalues:np.ndarray, row:np.ndarray)->np.ndarray:
    """
    If there are m features, and 1000 rows, then
    allValues = 1000 x m            and
    rows = m x 1
    """

    """
    From each row in allvalues, subtract the row
    Then square the difference
    This wil be a 1000 x m matrix
    """
    diff2 = np.square(allvalues - row)

    """
    For each row in diff2, sum all the columns, then apply sqrt
    and return.
    This will be 1000x1
    """
    return np.sqrt(np.sum(diff2, axis=1))




def read_file(filename:str) -> np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')




def generate_centroids(data:np.ndarray, k:int) -> np.ndarray:
    """
    Randomly choose k points as initial centroids
    """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices,:]




def assign_centroids(data:np.ndarray, centroids:np.ndarray)->list:
    arr = []
    """
    For each centroid
    """
    for centroid in centroids:
        """
        Calculate the data between this centroid and each of the data points
        This will give a kx1000 matrix
        """
        distances = calculate_distances(data, centroid)
        arr.append(distances)
    """
    Append k 1x1000 matrices to make a kx1000 matrix
    """
    arr = np.array(arr)

    """
    Find the index of the smallest centroid. If there are k centroids, then
    the distance of the array is kx1000.
    For each 1000 columns, find the row i that has the lowest value
    and that is the index of the smallest centroid, and that is what we assign
    to this point

    if there are k centroids, then all elements in this array lie between [0, k01]
    """
    indices = np.argmin(arr, axis=0)

    """
    Return a 1000x1 array
    """
    return indices.T



def calculate_error(data:np.ndarray, centroids:np.ndarray, assignments:np.ndarray)->float:
    """
    Assignments are the indices of the centroids that are closest to each point
    It is an array 1000x1
    We need to convert this to an array containing the actual centroids.
    If there are m features, then this would be a 1000xm array
    This is easily done by indexing
    """
    closest_centroids = centroids[assignments, :]

    """
    Subtract each axis of the closest centroid from each point
    """
    square_distances = np.square(np.subtract(data, closest_centroids))
    """
    return the mean of the square of the distances
    We can do some optimization here
    Distance of each point = SQRT(SUM_OVER_i((xi - yi)^2)), i = number of features
    Mean distance = [(Distance of each point) ^ 2] / m
    We can get rid of the square root and just add all the individual differences,
    for all the axes for all the points, and return it
    """
    return np.sum(square_distances) / data.shape[0]


def calculate_error2(data:np.ndarray, assignments:np.ndarray)->float:
    """
    This version of calculate_error calculates the centroids in case they are
    not already calculated
    """
    centroid_nums = np.unique(assignments)
    centroids = []
    for i in centroid_nums:
        pts_for_centroid = data[assignments == i]
        thecentroid = np.mean(pts_for_centroid, axis=0)
        centroids.append(thecentroid)
    return calculate_error(data, np.array(centroids), assignments)

def move_centroids(data:np.ndarray, assignments:np.ndarray, num_centroids:int)->np.ndarray:
    """
    If there are m features, and 1000 instances
    data = 1000 x m
    assignments = 1000x1

    where assignment is an array where each element a the index i
    is the number of the centroid that is closest to the element
    It takes a value between [0, k-1] if there are k centroids
    """
    new_centroids = []
    """
    For each centroid
    """
    for i in range(num_centroids):
        """
        Find the data points that are assigned to this centroid
        """
        pts_for_centroid = data[assignments == i]
        """
        Find the new centroid for all those data points
        by taking mean on each axis, this is 1xm if there are m axes, and
        append it to a list
        """
        new_centroids.append(np.mean(pts_for_centroid, axis=0))
    """
    Return all the new centroids, this is k x m
    """
    return np.array(new_centroids)




def iterate_knn(data:np.ndarray, num_centroids:int, iterations:int)->tuple:
    """
    Generate Random centroids
    """
    centroids = generate_centroids(data, num_centroids)
    for kk in range(iterations):
        """
        Get the centroid each data point is assigned to, this is an integer
        """
        assignments = assign_centroids(data, centroids)
        """
        Calculate the error, we're not stopping early yet, but we can
        use this to stop early if the error is not changing anymore
        """
        error = calculate_error(data, centroids, assignments)
        """
        Move Centroids, this will give us the new set of centroids
        if there are k centroids, and m features, this is k x m
        """
        centroids = move_centroids(data, assignments, num_centroids)
    """
    Calculate the final error, return this and along with the assigned
    centroids
    """
    error = calculate_error(data, centroids, assignments)
    return error, assignments




def restart_KMeans(filename:str, num_centroids:int, iterations:int, restarts:int, no_normalize:bool):
    data = read_file(filename)

    """
    Normalize the data
    """
    if not no_normalize:
        themean = np.mean(data, axis=0)
        stddev = np.std(data, axis=0)
        norm_data = (data - themean) / stddev

    best_error = None
    best_assignment = None
    """
    Run for N restarts
    """
    for i in range(restarts):
        if not no_normalize:
            error, assignments = iterate_knn(np.copy(norm_data), num_centroids, iterations)
            """
            The error is calculated based on the normalized data, so we must recalculate
            it based on the non-normalized data, otherwise the scale of the error
            will not match up with the non-normalized version
            """
            error = calculate_error2(data, assignments)
        else:
            error, assignments = iterate_knn(np.copy(data), num_centroids, iterations)
        if None == best_error or error < best_error:
            best_error = error
            best_assignment = assignments
    return best_error, best_assignment




def restart_and_elbow_plot(filename:str, iterations:int, restarts:int, max_N:int, no_normalize:bool):
    x = []
    y = []
    for i in range(3, max_N+1):
        error, assignment = restart_KMeans(filename, i, iterations, restarts, no_normalize)
        print(i, error)
        x.append(i)
        y.append(error)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    with open("plot_save.pickle", "wb") as f:
        pickle.dump(fig, f, pickle.HIGHEST_PROTOCOL)
    plt.show()




if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    parser.add_argument("-nn", "--no-normalize", help="Do not normalize data", action="store_true", default=False)
    args = parser.parse_args()
    restart_and_elbow_plot(args.file, 200, 10, 10, args.no_normalize)
