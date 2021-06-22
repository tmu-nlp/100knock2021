# 77. ミニバッチ化
# 問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．
# Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
'''
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
    '''
import time

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs):
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

  # 学習
  log_train = []
  log_valid = []
  for epoch in range(num_epochs):
    # 開始時刻の記録
    s_time = time.time()

    # 訓練モードに設定
    model.train()
    for inputs, labels in dataloader_train:
      # 勾配をゼロで初期化
      optimizer.zero_grad()

      # 順伝播 + 誤差逆伝播 + 重み更新
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # チェックポイントの保存
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

    # 終了時刻の記録
    e_time = time.time()

    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 

  return {'train': log_train, 'valid': log_valid}

# datasetの作成
dataset_train = CreateDataset(X_train, y_train)
dataset_valid = CreateDataset(X_valid, y_valid)

# モデルの定義
model = SLPNet(300, 4)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# モデルの学習
for batch_size in [2 ** i for i in range(11)]:
  print(f'バッチサイズ: {batch_size}')
  log = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, 1)
