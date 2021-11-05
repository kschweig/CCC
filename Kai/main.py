import numpy as np
import os
import matplotlib.pyplot as plt

data = []

with open(os.path.join("data", "level_1_1.csv"), "r") as f:

    for l, line in enumerate(f):
        if l == 0:
            N = int(line)
        elif l == 1:
            T = int(line)
        else:
            data.append([int(s) for s in line.split(",")])

data = np.asarray(data)
print(data.shape)