from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load(path):
    train_X, train_y = [], []
    with open(path, "r") as f:
        data = f.readlines()
    for line in data:
        line = line.strip().split(",")
        train_X.append(list(map(float, line[1:])))
        train_y.append(int(line[0]))
    return train_X, train_y

train_X, train_y = load("./data/train.feature.txt")
valid_X, valid_y = load("./data/valid.feature.txt")
test_X, test_y = load("./data/test.feature.txt")

Cs = np.arange(0.1, 5.1, 0.1)
lrs = [LogisticRegression(C=C, max_iter=1000).fit(train_X, train_y) for C in tqdm(Cs)]

def accuracy(lr, xs, ts):
    ys = lr.predict(xs)
    return (ys == ts).mean()

train_accs = [accuracy(lr, train_X, train_y) for lr in lrs]
valid_accs = [accuracy(lr, valid_X, valid_y) for lr in lrs]
test_accs = [accuracy(lr, test_X, test_y) for lr in lrs]

plt.plot(Cs, train_accs, label = 'train')
plt.plot(Cs, valid_accs, label = 'valid')
plt.plot(Cs, test_accs, label = 'test')
plt.legend()
plt.show()
