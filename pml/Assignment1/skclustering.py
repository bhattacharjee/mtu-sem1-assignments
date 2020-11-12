#!/usr/bin/env python3

import math
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

g_apply_scaling = False
g_apply_normalization = False
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def read_file(filename:str) -> np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')


def restart_KMeans(filename:str, num_centroids:int, iterations:int, restarts:int):
    data = read_file(filename)
    best_error = None
    best_assignment = None
    """
    Run for N restarts
    """
    for i in range(restarts):
        error, assignments = iterate_knn(np.copy(data), num_centroids, iterations)
        if None == best_error or error < best_error:
            best_error = error
            best_assignment = assignments
    return best_error, best_assignment

def get_centroids(data:np.ndarray, assignments:np.ndarray):
    centroids= []
    centroid_nums = np.unique(assignments)
    for i in centroid_nums:
        points = data[assignments == i,:]
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

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

def restart_and_elbow_plot_with_pca(filename:str, iterations:int, restarts:int, max_N:int, pca_value:int):
    x = []
    y = []
    data = read_file(filename)
    scaler = StandardScaler().fit(data.copy())
    scaled_data = scaler.transform(data.copy())
    scaled_save = scaled_data.copy()

    if 0 != pca_value:
        pca = PCA(n_components=pca_value)
        scaled_data = pca.fit_transform(scaled_data)

    for i in range(3, max_N+1):
        best_error = 99999999999
        best_assignment = None
        for j in range(5): # restarts
            clf = KMeans(n_clusters=i, random_state=0).fit(scaled_data)
            assignments = clf.predict(scaled_data).copy()
            centroids = clf.cluster_centers_
            scaled_centroids = get_centroids(scaled_save, assignments)
            error = calculate_error(scaled_save, scaled_centroids, assignments)
            if best_error > error:
                best_error = error
                best_assignment = assignments
        x.append(i)
        y.append(best_error)
        print(i, best_error)
    plt.plot(x, y, label=f'pca = {pca_value}')




if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    args = parser.parse_args()
    for i in range(6):
        restart_and_elbow_plot_with_pca(args.file, 200, 10, 10, i)
    plt.legend()
    plt.show()
