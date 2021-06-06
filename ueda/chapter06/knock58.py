import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
import pandas as pd

def train_logistic_param(regularization):
    df = pd.read_csv(r'C:\Git\train.feature.txt', encoding='utf-8')
    with open(r'C:\Git\train.txt', encoding="utf-8") as f:
        lr = LogisticRegression(random_state = 2021, max_iter=10000, C = regularization)
        lr.fit(df, [line.split('\t')[0] for line in f])
        return lr

if __name__ == "__main__":
    C=np.arange(0.1, 10.1, 0.5)
    accuracy = defaultdict(list)
    for regularization in C: 
        lr = train_logistic_param(regularization)
        for data in ['train', 'valid', 'test']:
            with open(r'C:\Git\{}.txt'.format(data), encoding="utf-8") as f:
                df = pd.read_csv(r'C:\Git\{}.feature.txt'.format(data), encoding='utf-8')
                cat = lr.predict(df)
                accuracy[data].append(accuracy_score([line.split('\t')[0] for line in f], cat))
    plt.plot(C, accuracy['train'], label='train')
    plt.plot(C, accuracy['valid'], label='valid')
    plt.plot(C, accuracy['test'], label='test')
    plt.legend()
    plt.show()
    