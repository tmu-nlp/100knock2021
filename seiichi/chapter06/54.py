from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd


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

print("train acc: {}, test acc: {}".format(accuracy(lr, train_X, train_y), accuracy(lr, test_X, test_y)))
# train acc: 0.9478659678023212, test acc: 0.9191616766467066
