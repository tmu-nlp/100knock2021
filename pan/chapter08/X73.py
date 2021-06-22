# 73.確率的勾配降下法による学習
# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ
# なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）

import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch import nn, optim

if __name__ == '__main__':
    X_train = joblib.load('X_train.joblib')
    y_train = joblib.load('y_train.joblib')
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

    model = nn.Linear(300, 4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)

    losses = []
    for epoch in tqdm(range(100)):
        optimizer.zero_grad() # 勾配をゼロで初期化
        y_pred = torch.softmax(model.forward(X_train), dim = -1)
        loss = loss_fn(y_pred, y_train)
        loss.backward() # 勾配を計算
        optimizer.step()
        losses.append(loss)

    # 結果を表示する
    print(model.state_dict()['weight'])