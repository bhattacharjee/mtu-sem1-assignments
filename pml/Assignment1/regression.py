#!/usr/local/bin/python3

import numpy as np
import argparse

def read_csv(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')


def calculate_distances(allvalues:np.ndarray, row:np.ndarray)->np.ndarray:
    diff2 = np.square(allvalues - row)
    return np.sqrt(np.sum(diff2, axis=1))

def get_min_max(allvalues:np.ndarray)->tuple:
    amin = np.amin(allvalues, axis=0)
    amax = np.amax(allvalues, axis=0)
    return amin, amax

def normalize(array:np.ndarray, amin:np.ndarray, amax:np.ndarray)->np.ndarray:
    x = array - amin
    scale = amax - amin
    return x / scale

def predict(train_features:np.ndarray, train_values:np.ndarray, test_features:np.ndarray, k:int, n:int)->np.ndarray:
    amin, amax = get_min_max(train_features)
    train_norm = normalize(train_features, amin, amax)
    test_norm = normalize(test_features, amin, amax)

    all_predictions = []

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


    for k in range(1, 20, 1):
        predicted = predict(train_features, train_labels, test_features, k, 1)
        r2 = calculate_r2(predicted, test_labels)
        print(k, r2)



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    args = parser.parse_args()
    
    file_name = args.file
    test_file_name = "./data/regression/testData.csv"

    main(file_name, test_file_name)
