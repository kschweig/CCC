import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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

"""
for i in range(5600):
    if i % 50 == 0:
        plt.imshow(images[i].reshape(28,-1))
        plt.title(f"label = {targets[i]}")
        plt.tight_layout()
        plt.savefig(os.path.join("task_2", f"label_{targets[i]}", f"{i}.png"))
"""

images = images / norm

images = images.reshape(len(images), 28, -1)
images = np.moveaxis(images, 1, 2)
plt.imshow(images[0].reshape(-1, 28))
plt.show()
raise ValueError

images = images.reshape(len(images), -1)

xtrain, xtest, ytrain, ytest = train_test_split(images, targets, test_size=0.1)

"""
pca = PCA(n_components=2)
xtrain = pca.fit_transform(xtrain)
"""

#regression = LogisticRegression(C=1e-1, max_iter=10000)
regression = RandomForestClassifier(max_depth=12)

"""
xtrain = pca.transform(xtrain)
xtest = pca.transform(xtest)
"""

regression.fit(xtrain, ytrain)

score = regression.score(xtest, ytest)

print(score)

if score < 0.9:
    raise ValueError


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

    images = images.reshape(len(images), 28, -1)
    images = np.moveaxis(images, 1, 2)
    images = images.reshape(len(images), -1)

    preds = regression.predict(images)

    with open(os.path.join("results", f"level_2_{c}.csv"), "w") as f:
        for r in preds:
            f.write(f"{int(r)}\n")



