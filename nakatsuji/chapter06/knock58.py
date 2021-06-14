path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#データ読み込み
X_train = pd.read_csv(path + 'train_feature.csv', sep='\t')
Y_train = pd.read_table(path + 'train.txt')["CATEGORY"]
X_valid = pd.read_csv(path + 'valid_feature.csv', sep='\t')
Y_valid = pd.read_table(path + 'valid.txt')["CATEGORY"]
X_test = pd.read_csv(path + 'test_feature.csv', sep='\t')
Y_test = pd.read_table(path + 'test.txt')["CATEGORY"]

C = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
train_acc = []
valid_acc = []
test_acc = []
for c in C:
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    train_acc.append(accuracy_score(Y_train, model.predict(X_train)))
    valid_acc.append(accuracy_score(Y_valid, model.predict(X_valid)))
    test_acc.append(accuracy_score(Y_test, model.predict(X_test)))

#グラフの表示（縦軸：acc　横軸：C)
plt.plot(C, train_acc, labed='train')
plt.plot(C, valid_acc, labed='valid')
plt.plot(C, test_acc, labed='test')
plt.xscale('log')
plt.ylabel("acc")
plt.xlabel('C')
plt.legend()
plt.show()