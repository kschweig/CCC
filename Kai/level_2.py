import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


images = []
targets = []


with open(os.path.join("data", f"train.csv"), "r") as f:

    for l, line in enumerate(f):
        if l == 0:
            N = int(line)
            print(N)
        elif l <= N:
            images.append([int(s) for s in line.split(",")])
        else:
            targets.append(int(line))

images = np.asarray(images)
print(N, images.shape, len(targets))

norm = np.max(images)

images = images / norm

xtrain, xtest, ytrain, ytest = train_test_split(images, targets, test_size=0.1)

#regression = LogisticRegression()
regression = RandomForestClassifier(max_depth=12)

regression.fit(xtrain, ytrain)

print(regression.score(xtest, ytest))


for c in range(1, 3):

    images = []

    with open(os.path.join("data", f"level_2_{c}.csv"), "r") as f:

        for l, line in enumerate(f):
            if l == 0:
                N = int(line)
                print(N)
            else:
                images.append([int(s) for s in line.split(",")])

    images = np.asarray(images)
    images = images / norm

    preds = regression.predict(images)

    with open(os.path.join("results", f"level_2_{c}.csv"), "w") as f:
        for r in preds:
            f.write(f"{int(r)}\n")



