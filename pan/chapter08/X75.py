# 75.損失と正解率のプロット
# 問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，学習の進捗状況を確認できるようにせよ．

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
    X_valid = joblib.load('X_valid.joblib')
    y_valid = joblib.load('y_valid.joblib')

    # numpyからtensorに変換する
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()
    X_valid = torch.from_numpy(X_valid.astype(np.float32)).clone()
    y_valid = torch.from_numpy(y_valid.astype(np.int64)).clone()

    # モデルを作成する
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(300, 4), nn.ReLU())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    max_train_acc = -1
    max_valid_acc = -1
    for epoch in tqdm(range(100000)):
        net.train()
        optimizer.zero_grad()
        y_pred = net(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        net.eval()
        # 損失を記録する
        train_losses.append(loss)
        valid_losses.append(loss_fn(net(X_valid), y_valid))
        # カテゴリを予測する
        values, y_pred_train = torch.max(net(X_train), 1)
        values, y_pred_valid = torch.max(net(X_valid), 1)
        # 正解率を求める
        train_acc = accuracy_score(y_pred_train, y_train)
        valid_acc = accuracy_score(y_pred_valid, y_valid)
        # 正解率を記録する
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        # 最大の正解率を記録する
        max_train_acc = max(max_train_acc, train_acc)
        max_valid_acc = max(max_valid_acc, valid_acc)

    # 最大の正解率をを表示する
    print(f'max train acc: {max_train_acc}')
    print(f'max valid acc: {max_valid_acc}')
    # 損失の変化をプロットする
    plt.plot(train_losses, label = 'train loss')
    plt.plot(valid_losses, label = 'valid loss')
    plt.legend()
    plt.show()
    # 正解率の変化をプロットする
    plt.plot(train_accs, label = 'train acc')
    plt.plot(valid_accs, label = 'valid acc')
    plt.legend()
    plt.show()