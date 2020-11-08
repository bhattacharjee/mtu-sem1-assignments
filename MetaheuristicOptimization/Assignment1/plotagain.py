#!/usr/bin/env python3
    """
    matplotlib plots that are saved as a pickle are re-rendered
    by this script.
    """

import matplotlib.pyplot as plt
import pickle
import sys
if len(sys.argv) < 2:
    print("plotagain.py filename1.pickle [filename2.pickle ...]")
    sys.exit(1)

args=sys.argv[1:]
for plot in args:
    ax = pickle.load(file=open(plot, "rb"))
plt.show()

