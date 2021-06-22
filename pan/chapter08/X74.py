#74.正解率の計測
#問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ

import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X_train = joblib.load('X_train.joblib')
    y_train = joblib.load('y_train.joblib')
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

    # testを読み込む
    X_test = joblib.load('X_test.joblib')
    y_test = joblib.load('y_test.joblib')
    X_test = torch.from_numpy(X_test.astype(np.float32)).clone()
    y_test = torch.from_numpy(y_test.astype(np.int64)).clone()

    net = nn.Linear(300, 4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    for epoch in range(10000):
        optimizer.zero_grad()
        y_pred = torch.softmax(net(X_train), dim=-1)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # trainの正解率を表示する
    _, y_pred_train = torch.max(net(X_train), 1)
    print(f'train acc: {accuracy_score(y_pred_train, y_train)}')

    # testの正解率を表示する
    _, y_pred_test = torch.max(net(X_test), 1)
    print(f'test  acc: {accuracy_score(y_pred_test, y_test)}')

# 結果
# train acc: 0.7828898050974513
# test  acc: 0.7856071964017991