#!/usr/bin/env python3

import numpy as np
import argparse

g_normalize_to_zero_mean_and_unit_variance = True
g_scale_between_zero_and_one = True

def read_csv(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')



def calculate_distances(allvalues:np.ndarray, row:np.ndarray)->np.ndarray:
    """
    Calculate the distance between all the training samples, and one
    point.

    If there are m training examples, and N features, then the shapes are as follows
    allvalues       m x N
    row             N x 1
    diff2           m x N
    np.sum(diff2)   m x 1
    Return          m x 1
    """
    diff2 = np.square(allvalues - row)
    return np.sqrt(np.sum(diff2, axis=1))



def get_min_max(allvalues:np.ndarray)->tuple:
    """
    If there are N features and m examples, then allvalues is
    m x N

    returns amin, amax, which are the min and max values feature wise.
    amin and amax are both
    Nx1
    """
    amin = np.amin(allvalues, axis=0)
    amax = np.amax(allvalues, axis=0)
    return amin, amax




def normalize(array:np.ndarray, stdarray:np.ndarray, meanarray:np.ndarray):
    global g_normalize_to_zero_mean_and_unit_variance
    if g_normalize_to_zero_mean_and_unit_variance:
        x = (array - meanarray) / stdarray
        return x
    else:
        return array




def scale(array:np.ndarray, amin:np.ndarray, amax:np.ndarray)->np.ndarray:
    global g_scale_between_zero_and_one
    if g_scale_between_zero_and_one:
        x = array - amin
        scale = amax - amin
        return x / scale
    else:
        return array




def predict(train_features:np.ndarray, train_values:np.ndarray, test_features:np.ndarray, k:int, n:int)->np.ndarray:
    global g_normalize_to_zero_mean_and_unit_variance, g_scale_between_zero_and_one
    if g_normalize_to_zero_mean_and_unit_variance:
        stddvarr = np.std(train_features, axis=0)
        meanarr = np.mean(train_features, axis=0)
        train_norm = normalize(train_features, stddvarr, meanarr) # Normalize
    else:
        train_norm = train_features

    if g_scale_between_zero_and_one:
        amin, amax = get_min_max(train_norm)
        train_norm = scale(train_norm, amin, amax)                # Scale

    if g_normalize_to_zero_mean_and_unit_variance:
        test_norm = normalize(test_features, stddvarr, meanarr)
    else:
        test_norm = test_features

    if g_scale_between_zero_and_one:
        test_norm = scale(test_norm, amin, amax)        # Normalize


    all_predictions = []

    """
    This version uses np.argsort(), and then chooses k values out of it
    The best case complexity of np.argsort() is O(n log(n))

    A better complexity can be achieved by using heapify.
    Heapify operation can run in O(n), and removing something from a heap is
    O(log(n))

    The algorithm can be as follows:
    1. Create a heap using the distances as the sort index in O(n) complexity
    2. Remove k elements from the heap, each in log(n) complexity
    The total complexity of the above is O(n + k*log(n))

    However, this scheme has not been implemented. For this size of data,
    the overheads might actually be higher than the speedup achieved.

    For larger data sets, this might be more efficient.
    """
    for i in test_norm:
        distances = calculate_distances(train_norm, i)
        sorted_indices = np.argsort(distances)[0:k]
        nearest_distances = distances[sorted_indices]
        nearest_values = train_values[sorted_indices]

        nearest_weights = 1 /  nearest_distances
        nearest_weights = nearest_weights ** n
        prediction = np.sum(np.multiply(nearest_weights, nearest_values))
        prediction = prediction / np.sum(nearest_weights)

        all_predictions.append(prediction)

    return all_predictions




def calculate_r2(predicted:np.ndarray, actual:np.ndarray)->float:
    sum_square_residuals = np.sum(np.square(actual - predicted))
    mean_actual = np.mean(actual)
    sum_squares = np.sum(np.square(actual - mean_actual))
    return 1 - (sum_square_residuals / sum_squares)



def main(filename:str, testfilename=str):
    array = read_csv(filename)
    test = read_csv(testfilename)

    train_features = array[:,:-1]
    train_labels = array[:,-1]
    test_features = test[:,:-1]
    test_labels = test[:,-1]


    for n in range(1, 3, 1):
        # Best values are k=10, n=5
        predicted = predict(train_features, train_labels, test_features, 3, n)
        r2 = calculate_r2(predicted, test_labels)
        print(n, r2)



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    parser.add_argument("-tf", "--test-file", help="Test file name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file
    test_file_name = args.test_file

    main(file_name, test_file_name)
