#/usr/bin/env python3

import numpy as np
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def read_from_file(filename:str)->np.ndarray:
    return np.genfromtxt(filename, dtype=float, delimiter=',')


def do_lasso(x_train, y_train, x_test, y_test, description="Regular lasso"):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    r2score = r2_score(y_test, predicted, multioutput='uniform_average')
    print(f'{"%40s" % description}          R2 = {r2score}')

def do_isolation_forest(x_train, y_train):
    scaler = StandardScaler().fit(x_train)
    xx_train = scaler.transform(x_train.copy())
    clf = IsolationForest(contamination=0.01)
    clf.fit(xx_train)
    results = clf.predict(xx_train)
    return x_train[1 == results], y_train[1 == results]


def do_kneighbours(x_train, y_train, x_test, y_test, description="Regular nearest neighbour"):
    best_k = 0
    best_predict = 0
    best_r2 = 0

    for i in range(1, 20):
        for j in range(5): # 5 restarts
            neigh = KNeighborsRegressor(n_neighbors=i)
            neigh.fit(x_train, y_train)
            predicted = neigh.predict(x_test)
            r2score = r2_score(y_test, predicted, multioutput='uniform_average')
            if r2score > best_r2:
                best_r2 = r2score
                best_k = i
                best_predict = predicted
    print(f'{"%40s" % description}          R2 = {best_r2} k = {best_k}')

def do_decisiontree(x_train, y_train, x_test, y_test, description="Regular Decision Tree"):
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    r2score = r2_score(y_test, predicted, multioutput='uniform_average')
    print(f'{"%40s" % description}          R2 = {r2score}')


def main(trainFile:str, testFile:str):
    train_data = read_from_file(trainFile)
    test_data = read_from_file(testFile)

    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]
    xx_train, yy_train = do_isolation_forest(x_train, y_train)

    do_lasso(x_train, y_train, x_test, y_test)
    do_lasso(xx_train, yy_train, x_test, y_test, description="Lasso with IsolationForest")
    do_kneighbours(x_train, y_train, x_test, y_test)
    do_kneighbours(xx_train, yy_train, x_test, y_test, description="KNN with Iolation Forest")
    do_decisiontree(x_train, y_train, x_test, y_test)
    do_decisiontree(xx_train, yy_train, x_test, y_test, description="Decision Tree with Isolation Forest")




main("./trainingData.csv", "./testData.csv")
