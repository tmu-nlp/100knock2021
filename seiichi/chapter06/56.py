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

tp = test_cm.diagonal()
tn = test_cm.sum(axis=1) - tp
fp = test_cm.sum(axis=0) - tp
p = tp / (tp + tn)
r = tp / (tp + fp)
F = 2 * p * r / (p + r)

micro_p = tp.sum() / (tp + tn).sum()
micro_r = tp.sum() / (tp + fp).sum()
micro_F = 2 * micro_p * micro_r / (micro_p + micro_r)
micro_ave = np.array([micro_p, micro_r, micro_F])

macro_p = p.mean()
macro_r = r.mean()
macro_F = 2 * macro_p * macro_r / (macro_p + macro_r)
macro_ave = np.array([macro_p, macro_r, macro_F])

categories = ['b', 't', 'e', 'm']

table = np.array([p, r, F]).T
table = np.vstack([table, micro_ave, macro_ave])
df = pd.DataFrame(
    table,
    index = categories + ['micro avg.'] + ['macro avg.'],
    columns = ['recall', 'precision', 'F1 score'])

print(df)

"""
              recall  precision  F1 score
b           0.952562   0.899642  0.925346
t           0.693252   0.856061  0.766102
e           0.980769   0.947635  0.963918
m           0.702703   0.962963  0.812500
micro avg.  0.919162   0.919162  0.919162
macro avg.  0.832321   0.916575  0.872419
"""
