import numpy as np
import os
import matplotlib.pyplot as plt

for c in range(1, 2):

    data = []

    with open(os.path.join("data", f"level_1_{c}.csv"), "r") as f:

        for l, line in enumerate(f):
            if l == 0:
                N = int(line)
            elif l == 1:
                T = int(line)
            else:
                data.append([int(s) for s in line.split(",")])

    data = np.asarray(data)

    data = np.mean(data, axis=1, where=data>0)
    results = []
    for i in data:
        if i > T:
            results.append(1)
        else:
            results.append(0)

    print(results)

    with open(os.path.join("results", f"level_1_{c}.csv"), "w") as f:
        f.write(f"{int(sum(results))}\n")
        for r in results:
            f.write(f"{int(r)}\n")
