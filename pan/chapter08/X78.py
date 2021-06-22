# 78. GPU上での学習
# 問題77のコードを改変し，GPU上で学習を実行せよ．

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import joblib
import torch

if __name__ == "__main__":
    # trainを読み込む
    X_train = joblib.load("X_train.joblib")
    y_train = joblib.load("y_train.joblib")
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

    # validを読み込む
    X_valid = joblib.load("X_valid.joblib")
    y_valid = joblib.load("y_valid.joblib")
    X_valid = torch.from_numpy(X_valid.astype(np.float32)).clone()
    y_valid = torch.from_numpy(y_valid.astype(np.int64)).clone()

    # モデルを作成する
    torch.manual_seed(0) # シードを固定する
    net = nn.Sequential(nn.Linear(300, 4), nn.PReLU())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    batchSize = [1, 2, 4, 8]

    ds = TensorDataset(X_train, y_train)

    X = X.to("cuda:0")
    y = y.to("cuda:0")
    net = net.to("cuda:0")

    for bs in batchSize:
        # 指定したバッチサイズにまとめたデータを順に取り出すことができる
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        for epoch in tqdm(range(10000)):
            train_running_loss = 0.0
            valid_running_loss = 0.0
            for xx, yy in loader:
                y_pred = net(xx)
                loss = loss_fn(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()
                valid_running_loss += loss_fn(net(X_valid), y_valid).item()

            joblib.dump(net.state_dict(), f"checkpoint_{epoch}.joblib")
            train_losses.append(train_running_loss)
            valid_losses.append(valid_running_loss)
            _, y_pred_train = torch.max(net(X_train), 1)
            train_accs.append(accuracy_score(y_pred_train, y_train))
            _, y_pred_valid = torch.max(net(X_valid), 1)
            valid_accs.append(accuracy_score(y_pred_valid, y_valid))

    plt.plot(train_losses, label="train loss")
    plt.plot(valid_losses, label="valid loss")
    plt.plot(train_accs, label="train acc")
    plt.plot(valid_accs, label="valid acc")
    plt.legend()
    plt.show()