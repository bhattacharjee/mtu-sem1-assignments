#!/usr/bin/env python3

import numpy as np
import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import seaborn as sns
import pandas as pd
#from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector

g_normalize_to_zero_mean_and_unit_variance = True
g_scale_between_zero_and_one = False

def read_csv(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')

def knn_regular(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, weights=wt)
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_mahalanobis(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='mahalanobis', metric_params={'V': np.cov(x_train.T)}, weights=wt)
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_seuclidean(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='seuclidean', metric_params={'V': np.cov(x_train.T)}, weights=wt)
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)

def knn_correlation(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
    r2scores = []
    indices = []
    for i in range(1, 20):
        neigh = KNeighborsRegressor(n_neighbors=i, metric='correlation', weights=wt)
        neigh.fit(x_train, y_train)
        predicted = neigh.predict(x_test)
        r2 = r2_score(y_test, predicted)
        r2scores.append(r2)
        indices.append(i)
        print(f"{description} - n={i} - r2 = {r2}")
    ax.plot(indices, r2scores, label=description)


def knn_pca(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
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
            neigh = KNeighborsRegressor(n_neighbors=i, metric="mahalanobis", metric_params={'V': np.cov(x_train_pca.T)}, weights=wt)
            neigh.fit(x_train_pca, y_train)
            predicted = neigh.predict(x_test_pca)
            r2 = r2_score(y_test, predicted)
            r2scores.append(r2)
            indices.append(i)
            print(f"{description} - pca={j} - n={i} - r2 = {r2}")
        the_description = f"{description}={j}"
        ax.plot(indices, r2scores, label=the_description)



def forward_selection(x_train, y_train, x_test, y_test, fig, ax, description, wt='distance'):
    for j in range(3, x_train.shape[1]):
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
        sfs = SequentialFeatureSelector(
                knn,
                k_features=j,
                forward=True,
                floating=False,
                scoring='r2',
                verbose=2)
        kk = sfs.fit(x_train, y_train)
        selected = [int(i) for i in sfs.k_feature_names_]
        print(selected)
        new_train_x = x_train[:,selected]
        new_test_x = x_test[:,selected]
        print(new_train_x.shape, new_test_x.shape)
        knn_regular(new_train_x, y_train, new_test_x, y_test, fig, ax, f"FS={sfs.k_feature_names_}", wt)



def main(filename:str, testfilename:str, mahalanobis:bool, pca:bool, correlation:bool):
    array = read_csv(filename)
    test = read_csv(testfilename)

    x_train = array[:,:-1]
    y_train = array[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    fig, ax = plt.subplots(1, 1)

    forward_selection(x_train, y_train, x_test, y_test, fig, ax, "FeatureSelect")

    knn_regular(x_train, y_train, x_test, y_test, fig, ax, description="Regular KNN distance weighted")
    """
    #knn_regular(x_train, y_train, x_test, y_test, fig, ax, description="Regular KNN uniform weighted", wt='uniform')
    if mahalanobis:
        knn_mahalanobis(x_train, y_train, x_test, y_test, fig, ax, description="Mahalanobis Distance distance weighted")
        knn_mahalanobis(x_train, y_train, x_test, y_test, fig, ax, description="Mahalanobis Distance uniform weighted", wt='uniform')
    if pca:
        knn_pca(x_train, y_train, x_test, y_test, fig, ax, description="pca")
    if correlation:
        knn_correlation(x_train, y_train, x_test, y_test, fig, ax, description="correlation distance weighted", wt='distance')
        knn_correlation(x_train, y_train, x_test, y_test, fig, ax, description="correlation uniform weighted", wt='uniform')

    # For some reason, knn_seuclidean doesn't work and python itself dumps core.
    #knn_seuclidean(x_train, y_train, x_test, y_test, fig, ax, description="SEuclidean")
    """
    fig.legend()
    #sns.pairplot(pd.DataFrame(x_train))

    with open("save.plot.pickle", "wb") as f:
        pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

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
