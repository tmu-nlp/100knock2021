from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import numpy as np

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
test_X, test_y = load("./data/test.feature.txt")

with open("bin/lr.model", "rb") as f:
    lr = pickle.load(f)

def predict(x, lr):
    out = lr.predict_proba(x)
    preds = out.argmax(axis=1)
    probs = out.max(axis=1)
    return preds, probs

def accuracy(lr, xs, ts):
    ys = lr.predict(xs)
    return (ys == ts).mean()

def confusion_matrix(xs, ts):
    num_class = np.unique(ts).size
    mat = np.zeros((num_class, num_class), dtype=np.int32)
    ys = lr.predict(xs)
    for y, t in zip(ys, ts):
        mat[t, y] += 1
    return mat

train_cm = confusion_matrix(train_X, train_y)
test_cm = confusion_matrix(test_X, test_y)
print(train_cm, test_cm)


"""
[[4459   39   55    4]
 [ 169  942   87    5]
 [  26    6 4148    0]
 [  79    6   81  578]]

[[502  14  11   0]
 [ 37 113  11   2]
 [  9   2 561   0]
 [ 10   3   9  52]]
"""
