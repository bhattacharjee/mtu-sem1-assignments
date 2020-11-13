#!/usr/bin/env python3

import numpy as np
import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

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
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_mahalanobis(x_train, y_train, x_test, y_test, fig, ax, description):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='mahalanobis', metric_params={'V': np.cov(x_train.T)})
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_seuclidean(x_train, y_train, x_test, y_test, fig, ax, description):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='seuclidean', metric_params={'V': np.cov(x_train.T)})
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_correlation(x_train, y_train, x_test, y_test, fig, ax, description):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='correlation')
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)


def knn_pca(x_train, y_train, x_test, y_test, fig, ax, description):
    for j in range(2, x_train.shape[1]):
        if j != (x_train.shape[1] -1):
            pca = PCA(n_components=j)
            pca.fit(x_train)
            x_train_pca = pca.transform(x_train)
            x_test_pca = pca.transform(x_test)
        else:
            x_train_pca = x_train
            x_test_pca = x_test
        r2scores = []
        indices = []
        for i in range(1, 20):
            neigh = KNeighborsRegressor(n_neighbors=i, metric="mahalanobis", metric_params={'V': np.cov(x_train_pca.T)})
            neigh.fit(x_train_pca, y_train)
            predicted = neigh.predict(x_test_pca)
            r2 = r2_score(y_test, predicted)
            r2scores.append(r2)
            indices.append(i)
            print(f"{description} - pca={j} - n={i} - r2 = {r2}")
        the_description = f"{description}={j}"
        ax.plot(indices, r2scores, label=the_description)

def main(filename:str, testfilename:str, mahalanobis:bool, pca:bool, correlation:bool):
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
    if mahalanobis:
        knn_mahalanobis(x_train, y_train, x_test, y_test, fig, ax, description="Mahalanobis Distance")
    if pca:
        knn_pca(x_train, y_train, x_test, y_test, fig, ax, description="pca")
    if correlation:
        knn_correlation(x_train, y_train, x_test, y_test, fig, ax, description="correlation")

    # For some reason, knn_seuclidean doesn't work and python itself dumps core.
    #knn_seuclidean(x_train, y_train, x_test, y_test, fig, ax, description="SEuclidean")
    fig.legend()

    plt.show()


# ----------------------------------------------------------------------

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name", type=str, required=True)
    parser.add_argument("-tf", "--test-file", help="Test file name", type=str, required=True)
    parser.add_argument("-m", "--mahalanobis", help="Use Mahalanobis Distance", action="store_true")
    parser.add_argument("-pca", "--use-pca", help="Use PCA with Mahalanobis Distance", action="store_true")
    parser.add_argument("-c", "--use-correlation-metric", help="Use PCA with correlation metric", action="store_true")
    args = parser.parse_args()

    file_name = args.file
    test_file_name = args.test_file

    main(file_name, test_file_name, args.mahalanobis, args.use_pca, args.use_correlation_metric)
