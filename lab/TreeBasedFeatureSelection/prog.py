#!/usr/bin/env python3

import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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


if "__main__" == __name__:
    main("./dataFile.csv")
