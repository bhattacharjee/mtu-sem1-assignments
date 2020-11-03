#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import sys

ax = pickle.load(file=open(sys.argv[1], "rb"))
plt.show()

