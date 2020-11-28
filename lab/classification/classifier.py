#/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def read_from_file(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')


def knn_classifier(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray):
    best_f1 = 0
    best_k = 0
    for i in range(2,20):
        for j in range(0, 10): # restarts
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(x_train, y_train)
            predicted = neigh.predict(x_test)
            f1 = f1_score(y_test, predicted)
            if (f1 > best_f1):
                best_f1 = f1
            best_k = i
    print(f"KNN:                Best f1 = {f1} for k = {best_k}")


def naive_bayes(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray):
    gnb = GaussianNB()
    predicted = gnb.fit(x_train, y_train).predict(x_test)
    f1 = f1_score(y_test, predicted)
    print(f"Gaussian NB:        Best f1 = {f1}")

def svm_classify(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray):
    svc = SVC()
    predicted = svc.fit(x_train, y_train).predict(x_test)
    f1 = f1_score(y_test, predicted)
    print(f"SVM Classification: Best f1 = {f1}")

def main(trainFile:str, testFile:str):
    train_data = read_from_file(trainFile)
    test_data = read_from_file(testFile)

    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    knn_classifier(x_train, y_train, x_test, y_test)
    naive_bayes(x_train, y_train, x_test, y_test)
    svm_classify(x_train, y_train, x_test, y_test)




main("./trainingData2.csv", "./testData2.csv")
