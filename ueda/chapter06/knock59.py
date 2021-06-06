import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
import pandas as pd

def train_logistic_param(norm, solver_name, ratio):
    df = pd.read_csv(r'C:\Git\train.feature.txt', encoding='utf-8')
    with open(r'C:\Git\train.txt', encoding="utf-8") as f:
        lr = LogisticRegression(random_state = 2021, max_iter=10000, penalty = norm, C=5, solver = solver_name , l1_ratio = ratio)
        lr.fit(df, [line.split('\t')[0] for line in f])
        return lr

def train_SVC(C_value, gamma_value):
    df = pd.read_csv(r'C:\Git\train.feature.txt', encoding='utf-8')
    with open(r'C:\Git\train.txt', encoding="utf-8") as f:
        clf = SVC(random_state = 2021, C=C_value, gamma = gamma_value)
        clf.fit(df, [line.split('\t')[0] for line in f])
        return clf

'''
for norm in ['none', 'l2']:
    with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
        df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
        lr = train_logistic_param(norm)
        cat = lr.predict(df)
        print((accuracy_score([line.split('\t')[0] for line in f], cat)))
'''
#penalty
#None = 0.897378277153558
#l2 = 0.9228464419475655

'''
for tol_value in [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
    with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
        df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
        lr = train_logistic_param('l2', tol_value)
        cat = lr.predict(df)
        print((accuracy_score([line.split('\t')[0] for line in f], cat)))
'''
#tol
#1e4 = 0.43745318352059925
#1e3 = 0.43745318352059925
#1e2 = 0.8456928838951311
#1e1 = 0.9191011235955057
#1e0 = 0.9220973782771535
#1e-1 = 0.9228464419475655
#1e-2 = 0.9228464419475655
#1e-3 = 0.9228464419475655
#1e-4 = 0.9228464419475655
'''
with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
    df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
    lr = train_logistic_param('l2', 'newton-cg')
    cat = lr.predict(df)
    print((accuracy_score([line.split('\t')[0] for line in f], cat)))
'''
#newton-cg = 0.9228464419475655

'''
with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
    df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
    lr = train_logistic_param('elasticnet', 'saga', 0.5)
    cat = lr.predict(df)
    print((accuracy_score([line.split('\t')[0] for line in f], cat)))
'''
#elasticnet = 0.9191011235955057

for C in [1, 10, 100]:
    for gamma in [0.1, 1, 10]:
        with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
            df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
            clf = train_SVC(C, gamma)
            cat = clf.predict(df)
            print((accuracy_score([line.split('\t')[0] for line in f], cat)))
#C=1, gamma=0.1 = 0.8651685393258427
#C=1, gamma=1 = 0.9123595505617977
#C=1, gamma=10 = 0.49962546816479403
#C=10, gamma=0.1 = 0.9176029962546817
#C=10, gamma=1 = 0.9205992509363295
#C=10, gamma=10 = 0.5213483146067416
#C=100, gamma=0.1 = 0.9086142322097378
#C=100, gamma=1 = 0.9205992509363295
#C=100, gamma=10 = 0.5213483146067416