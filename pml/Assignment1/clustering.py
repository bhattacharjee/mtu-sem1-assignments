#!/usr/bin/env python3

import math
import numpy as np
import argparse
import sys

def calculate_distances(allvalues:np.ndarray, row:np.ndarray)->np.ndarray:
    diff2 = np.square(allvalues - row)
    return np.sqrt(np.sum(diff2, axis=1))

def read_file(filename:str) -> np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')

def generate_centroids(data:np.ndarray, k:int) -> np.ndarray:
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices,:]

def assign_centroids(data:np.ndarray, centroids:np.ndarray)->list:
    arr = []
    for centroid in centroids:
        distances = calculate_distances(data, centroid)
        arr.append(distances)
    arr = np.array(arr)
    indices = np.argmin(arr, axis=0)
    return indices.T

def calculate_error(data:np.ndarray, centroids:np.ndarray, closest_centroid_indices:np.ndarray)->float:
    distance = 0
    closest_centroids = centroids[closest_centroid_indices, :]
    square_distances = np.square(np.subtract(data, closest_centroids))
    return np.sum(square_distances) / data.shape[0]

def move_centroids(data:np.ndarray, closest_centroids:np.ndarray, num_centroids:int)->np.ndarray:
    new_centroids = []
    for i in range(num_centroids):
        pts_for_centroid = data[closest_centroids == i]
        new_centroids.append(np.mean(pts_for_centroid, axis=0))
    return np.array(new_centroids)

def main(filename:str, num_centroids:int, iterations:int):
    data = read_file(filename)
    centroids = generate_centroids(data, num_centroids)
    for kk in range(iterations):
        closest_centroids = assign_centroids(data, centroids)
        error = calculate_error(data, centroids, closest_centroids)
        print(error)
        centroids = move_centroids(data, closest_centroids, num_centroids)

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    args = parser.parse_args()
    main(args.file, num_centroids=3, iterations=100)
