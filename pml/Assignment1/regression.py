#!/usr/local/bin/python3

import numpy as np
import argparse

def read_csv(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')


def calculate_distances(allvalues:np.ndarray, row:np.ndarray)->np.ndarray:
    diff2 = np.square(allvalues - row)
    return np.sum(diff2, axis=1)

def get_min_max(allvalues:np.ndarray)->tuple:
    amin = np.amin(allvalues, axis=0)
    amax = np.amax(allvalues, axis=0)
    return amin, amax

def normalize(array:np.ndarray, amin:np.ndarray, amax:np.ndarray)->np.ndarray:
    x = array - amin
    scale = amax - amin
    return x / scale
    
def main(filename:str, testfilename=str):
    array = read_csv(filename)
    test = read_csv(testfilename)
    calculate_distances(array, array[0])
    amin, amax = get_min_max(array)
    train_norm = normalize(array, amin, amax)
    test_norm = normalize(test, amin, amax)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    args = parser.parse_args()
    
    file_name = args.file
    test_file_name = "./data/regression/testData.csv"

    main(file_name, test_file_name)
