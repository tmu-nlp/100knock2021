#knock75
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import numpy as np


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1) #各単語ベクトルの要素の最大値
  label = label.data.numpy() #ラベルをpredと比べられる形にしてる?
  return (pred == label).mean() #正解率


X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')

model = LogisticRegression() #モデルの定義
ds = TensorDataset(X_train, y_train) #Datasetの定義
loader = DataLoader(ds, batch_size=1, shuffle=True) # DataLoaderを作成
loss_fn = torch.nn.CrossEntropyLoss() #損失関数の定義
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1) #オプティマイザの定義

log_train = []
log_valid = []
for epoch in tqdm(range(50)):
    for xx, yy in loader:
        optimizer.zero_grad() #勾配を0で初期化
        y_pred = model(xx) #categoryの予測
        loss = loss_fn(y_pred, yy) #損失率計算
        loss.backward() #誤差逆伝播
        optimizer.step() #重み更新
    with torch.no_grad():
      y_pred = model(X_train)
      loss = loss_fn(y_pred, y_train)
      acc = accuracy(y_pred, y_train)
      log_train.append([loss, acc])

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      acc = accuracy(y_pred, y_valid)
      log_valid.append([loss, acc])

# 視覚化
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(np.array(log_train).T[0], label='train')
ax[0].plot(np.array(log_valid).T[0], label='valid')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[1].plot(np.array(log_train).T[1], label='train')
ax[1].plot(np.array(log_valid).T[1], label='valid')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()
plt.show()