#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def read_file(filename:str)->np.ndarray:
    data = np.genfromtxt(filename, delimiter=',')
    return data


def main(filename:str):
    data = read_file(filename)
    print(data.shape)

    # Split the data into the features and labels
    X = data[:,:-1]
    y = data[:,-1]

    # Now standardize the data
    scaler = StandardScaler().fit(X)
    stdX = scaler.transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(stdX, y, test_size=0.2, random_state=42)

    # Create a KNN and test the accuracy
    knclassifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knclassifier.fit(X_train, y_train)
    predicted = knclassifier.predict(X_test)
    f1score = f1_score(y_test, predicted)
    accuracy = accuracy_score(y_test, predicted)
    print(f"f1 = {f1score} accuracy = {accuracy}")

    # Get the most import metrics
    rfc = RandomForestClassifier().fit(X_train, y_train)
    feature_importances = np.argsort(rfc.feature_importances_)

    indices = []
    f1_scores = []
    accuracy_scores = []
    for i in range(1, len(feature_importances)):
        features = feature_importances[i:]
        new_X_train = X_train[:, features]
        new_X_test = X_test[:, features]
        knnclassifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knnclassifier.fit(new_X_train, y_train)
        predicted = knnclassifier.predict(new_X_test)
        f1score = f1_score(y_test, predicted)
        accuracy = accuracy_score(y_test, predicted)
        indices.append(i)
        f1_scores.append(f1score)
        accuracy_scores.append(accuracy)

    fig, ax = plt.subplots(1, 1)
    ax.plot(indices, f1_scores, label="f1_score")
    ax.plot(indices, accuracy_scores, label="accuracy")
    fig.legend()
    plt.show()


if "__main__" == __name__:
    main("./dataFile.csv")
