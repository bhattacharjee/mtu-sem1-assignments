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
    """
    diff2 is m x N, summing it along axis 1 will give an m x 1 array
    """
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
    """
    Normalize to zero mean and unit standard deviation

    If there are m training examples and N features
    array       m x N
    sdarray     N x 1
    meanarray   N x 1
    """
    global g_normalize_to_zero_mean_and_unit_variance
    if g_normalize_to_zero_mean_and_unit_variance:
        x = (array - meanarray) / stdarray
        return x
    else:
        return array




def scale(array:np.ndarray, amin:np.ndarray, amax:np.ndarray)->np.ndarray:
    """
    Scale so that all values are between 0 and 1

    If there are m training examples and N features
    array       m x N
    amin        N x 1
    amax        N x 1
    """
    global g_scale_between_zero_and_one
    if g_scale_between_zero_and_one:
        x = array - amin
        scale = amax - amin
        return x / scale
    else:
        return array




def predict(train_features:np.ndarray, train_values:np.ndarray, test_features:np.ndarray, k:int, p:int)->np.ndarray:
    global g_normalize_to_zero_mean_and_unit_variance, g_scale_between_zero_and_one

    if g_normalize_to_zero_mean_and_unit_variance:
        """
        If normalization is required, normalize to zero mean and unit
        standard deviation.
        """
        stddvarr = np.std(train_features, axis=0)
        meanarr = np.mean(train_features, axis=0)
        train_norm = normalize(train_features, stddvarr, meanarr) # Normalize
    else:
        """
        If normalization is not required do nothing
        """
        train_norm = train_features

    if g_scale_between_zero_and_one:
        """
        If scaling is required scale to a range in [0, 1]
        """
        amin, amax = get_min_max(train_norm)
        train_norm = scale(train_norm, amin, amax)                # Scale

    if g_normalize_to_zero_mean_and_unit_variance:
        """
        Normalize the test values by the same amount we had normalized the
        training features.
        It is important to use the values of stddvarr and meanarr exactly
        as we had used in the training normalization
        """
        test_norm = normalize(test_features, stddvarr, meanarr)
    else:
        test_norm = test_features

    if g_scale_between_zero_and_one:
        """
        Scale the test features exactly as we had scaled the train features
        Again, it is important to use the same values of amin and amax that we
        had used in the training scaling
        """
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
        """
        For each data point in the test data, categorize it
        """

        """
        Calculate the idstance between this point, and all other points
        """
        distances = calculate_distances(train_norm, i)

        """
        Sort based on the distance, and get the k smallest indices
        """
        sorted_indices = np.argsort(distances)[0:k]

        """
        Get the nearest distances, this will be used in the weighted average
        """
        nearest_distances = distances[sorted_indices]

        """
        Avoid division by zero
        """
        nearest_distances = nearest_distances + 0.0000000001

        """
        Get the y for the nearest k points
        """
        nearest_values = train_values[sorted_indices]

        """
        Weight is inverse of distance
        """
        nearest_weights = 1 /  nearest_distances

        """
        The power used in calculating the weight is a parameter
        weight = (1/distance) ^ p
        """
        nearest_weights = nearest_weights ** p

        """
        Multiply each of the nearest weights with the nearest y values
        and sum it up
        """
        prediction = np.sum(np.multiply(nearest_weights, nearest_values))

        """
        Divide by the sum of the weights so that everything adds up and the
        weighted average is proper
        """
        prediction = prediction / np.sum(nearest_weights)

        """
        Add this prediction to an array, so that we can get all the predictions
        for all the points in one place and return it
        """
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


    print("\n")
    print("Iterate through different powers for k=3")
    print("POWER                    ERROR")
    for power in range(0, 20, 1):
        # Best values are k=10, n=5
        predicted = predict(train_features, train_labels, test_features, 3, power)
        r2 = calculate_r2(predicted, test_labels)
        print("%0.2d                     " % (power,), r2)

    print("\n")
    print("Best value is calculated for k=10 and n=5")
    predicted = predict(train_features, train_labels, test_features, 10, 5)
    r2 = calculate_r2(predicted, test_labels)
    print("POWER                    ERROR")
    print("%0.2d                     " % (5,), r2)



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    parser.add_argument("-tf", "--test-file", help="Test file name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file
    test_file_name = args.test_file

    main(file_name, test_file_name)
