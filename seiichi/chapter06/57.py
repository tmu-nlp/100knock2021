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

categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']
with open("./data/vocab.txt") as f:
    vocab = [v.strip() for v in f.readlines()]

def show_weight(directional, N):
    for i, cat in enumerate(categories):
        indices = lr.coef_[i].argsort()[::directional][:N]
        best = np.array(vocab)[indices]
        weight = lr.coef_[i][indices]
        print(category_names[i])
        print(pd.DataFrame([best, weight], index = ['特徴量', '重み'], columns = np.arange(N) + 1))

show_weight(-1, 10)
show_weight(1, 10)

"""
business
          1        2        3        4        5       6        7        8        9       10
特徴量     bank      fed    china   profit      ecb  ukrain     euro      oil      low     ceo
重み   3.73495  3.22488  2.96542  2.74345  2.70994  2.6685  2.60012  2.52548  2.48829  2.4717
science and technology
          1         2        3        4          5        6        7        8        9        10
特徴量    googl  facebook     appl   climat  microsoft    tesla       gm     nasa  comcast     moon
重み   5.10463   4.72997  3.86494  3.74098    3.30935  3.00725  2.92823  2.47085  2.43093  2.40143
entertainment
             1       2        3        4        5       6        7        8        9        10
特徴量  kardashian   chris     star     film     movi     kim    miley      her    cyrus      fan
重み      3.20181  2.8137  2.70361  2.64176  2.55453  2.4688  2.36357  2.34002  2.31183  2.27793
health
          1        2        3        4        5        6         7        8        9        10
特徴量    ebola     drug   cancer      fda    studi      mer  cigarett    brain   doctor   health
重み   4.26151  3.63596  3.45076  3.27091  3.21988  2.97341   2.56911  2.52664  2.41868  2.37216

business
          1         2        3        4        5        6           7        8        9       10
特徴量      her  facebook    video     star    ebola        !  kardashian    googl    virus    babi
重み  -1.91393  -1.87009 -1.68673 -1.67657 -1.67632 -1.60403    -1.56851 -1.56538 -1.50635 -1.5024
science and technology
          1        2        3       4        5         6         7           8         9         10
特徴量     drug      fed   cancer    rate      her       low       his  kardashian       oil    throne
重み  -1.34925 -1.07557 -1.05303 -1.0518 -1.00303 -0.995201 -0.982457   -0.934464 -0.927487 -0.920039
entertainment
          1      2        3        4        5         6        7       8        9        10
特徴量    googl  china      ceo        1      buy  facebook     rise    sale       gm    studi
重み  -2.49559 -2.269 -2.25247 -2.15858 -2.06084  -2.01532 -2.00499 -1.9739 -1.88899 -1.88637
health
          1        2        3        4        5         6         7         8         9         10
特徴量     deal      ceo    googl     bank       gm      appl    climat  facebook       buy      sale
重み  -1.09346 -1.07793 -1.04365 -1.03528 -1.02794 -0.906962 -0.873965 -0.844569 -0.797558 -0.779836
"""
