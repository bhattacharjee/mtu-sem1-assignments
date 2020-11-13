#!/usr/bin/env python3

import numpy as np
import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

g_normalize_to_zero_mean_and_unit_variance = True
g_scale_between_zero_and_one = False

def read_csv(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')

def knn_regular(x_train, y_train, x_test, y_test, fig, ax, description):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i)
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
    ax.plot(indices, r2scores, label=description)


def main(filename:str, testfilename=str):
    array = read_csv(filename)
    test = read_csv(testfilename)

    x_train = array[:,:-1]
    y_train = array[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    fig, ax = plt.subplots(1, 1)

    knn_regular(x_train, y_train, x_test, y_test, fig, ax, description="Regular KNN")
    fig.legend()

    plt.show()


# ----------------------------------------------------------------------

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    parser.add_argument("-tf", "--test-file", help="Test file name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file
    test_file_name = args.test_file

    main(file_name, test_file_name)
