from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

def accuracy(lr, xs, ts):
    ys = lr.predict(xs)
    return (ys == ts).mean()

params = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
clf = GridSearchCV(LogisticRegression(max_iter=1000), params)
clf.fit(train_X, train_y)

print(clf.cv_results_)

print(accuracy(clf, test_X, test_y))


"""
{'mean_fit_time': array([ 5.04500561,  5.55758481,  1.50603561, 25.07627587, 17.8504056 ]), 'std_fit_time': array([0.15790421, 0.6110677 , 0.01631593, 2.71416458, 0.48478533]), 'mean_score_time': array([0.36157255, 0.3556438 , 0.33979836, 0.32718458, 0.31889362]), 'std_score_time': array([0.01983028, 0.00335945, 0.00256027, 0.01119065, 0.00230713]), 'param_solver': masked_array(data=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], mask=[False, False, False, False, False], fill_value='?', dtype=object), 'params': [{'solver': 'newton-cg'}, {'solver': 'lbfgs'}, {'solver': 'liblinear'}, {'solver': 'sag'}, {'solver': 'saga'}], 'split0_test_score': array([0.87693028, 0.87693028, 0.86757136, 0.87693028, 0.87693028]), 'split1_test_score': array([0.90032756, 0.90032756, 0.88909686, 0.90032756, 0.90032756]), 'split2_test_score': array([0.88816097, 0.88816097, 0.87786617, 0.88816097, 0.88816097]), 'split3_test_score': array([0.88769303, 0.88769303, 0.88067384, 0.88769303, 0.88769303]), 'split4_test_score': array([0.89419476, 0.89419476, 0.8829588 , 0.89419476, 0.89419476]), 'mean_test_score': array([0.88946132, 0.88946132, 0.87963341, 0.88946132, 0.88946132]), 'std_test_score': array([0.00777593, 0.00777593, 0.00707551, 0.00777593, 0.00777593]), 'rank_test_score': array([1, 1, 5, 1, 1], dtype=int32)}
0.9191616766467066
"""
