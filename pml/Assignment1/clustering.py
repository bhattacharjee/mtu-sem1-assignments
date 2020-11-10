#!/usr/bin/env python3

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

def move_centroids(data:np.ndarray, closest_centroids:np.ndarray, num_centroids:int)->np.ndarray:
    new_centroids = []
    for i in range(num_centroids):
        print(closest_centroids.shape, data.shape)
        pts_for_centroid = data[closest_centroids == i]
        new_centroids.append(np.mean(pts_for_centroid, axis=0))
    return np.array(new_centroids).shape

def main(filename:str):
    data = read_file(filename)
    centroids = generate_centroids(data, 3)
    closest_centroids = assign_centroids(data, centroids)
    centroids = move_centroids(data, closest_centroids, 3)

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    args = parser.parse_args()
    main(args.file)
