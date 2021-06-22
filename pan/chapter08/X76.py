# 76. チェックポイント
# 問題75のコードを改変し，各エポックのパラメータ更新が完了するたびに，チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）をファイルに書き出せ．

import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X_train = joblib.load('X_train.joblib')
    y_train = joblib.load('y_train.joblib')
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

    # validを読み込む
    X_valid = joblib.load('X_valid.joblib')
    y_valid = joblib.load('y_valid.joblib')
    X_valid = torch.from_numpy(X_valid.astype(np.float32)).clone()
    y_valid = torch.from_numpy(y_valid.astype(np.int64)).clone()

    # モデルを作成する
    torch.manual_seed(0) # シードを固定する
    net = nn.Sequential(nn.Linear(300, 4), nn.PReLU())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in tqdm(range(100000)):
        net.train()
        optimizer.zero_grad() # 勾配をゼロで初期化する
        y_pred = net(X_train) # ラベル予測する
        loss = loss_fn(y_pred, y_train) # 損失を求める
        loss.backward() # 勾配を計算
        optimizer.step()

        # モデルを保存する
        joblib.dump(net.state_dict(), f'checkpoint_{epoch}.joblib')

        net.eval()
        train_losses.append(loss)
        valid_losses.append(loss_fn(net(X_valid), y_valid))
        # values, indices = torch.max(tensor, 0)
        values, y_pred_train = torch.max(net(X_train), 1)
        values, y_pred_valid = torch.max(net(X_valid), 1)
        train_accs.append(accuracy_score(y_pred_train, y_train))
        valid_accs.append(accuracy_score(y_pred_valid, y_valid))

    plt.plot(train_losses, label = 'train loss')
    plt.plot(valid_losses, label = 'valid loss')
    plt.legend()
    plt.show()

    plt.plot(train_accs, label = 'train acc')
    plt.plot(valid_accs, label = 'valid acc')
    plt.legend()
    plt.show()