# 79. 多層ニューラルネットワーク
# 問題78のコードを改変し，バイアス項の導入や多層化など，
# ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．
'''
import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

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

    net = nn.Sequential(
        nn.Linear(300, 100),
        nn.PReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 25),
        nn.PReLU(),
        nn.BatchNorm1d(25),
        nn.Linear(25, 4)
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    batchSize = [64]

    X = X.to('cuda:0')
    y = y.to('cuda:0')
    net = net.to('cuda:0')

    ds = TensorDataset(X, y)

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

            joblib.dump(net.state_dict(), f'checkpoint_{epoch}.joblib')
            train_losses.append(train_running_loss)
            valid_losses.append(valid_running_loss)
            _, y_pred_train = torch.max(net(X_train), 1)
            train_accs.append(accuracy_score(y_pred_train, y_train))
            _, y_pred_valid = torch.max(net(X_valid), 1)
            valid_accs.append(accuracy_score(y_pred_valid, y_valid))

    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='valid loss')
    plt.plot(train_accs, label='train acc')
    plt.plot(valid_accs, label='valid acc')
    plt.legend()
    plt.show()
    '''
from torch.nn import functional as F

class MLPNet(nn.Module):
  def __init__(self, input_size, mid_size, output_size, mid_layers):
    super().__init__()
    self.mid_layers = mid_layers
    self.fc = nn.Linear(input_size, mid_size)
    self.fc_mid = nn.Linear(mid_size, mid_size)
    self.fc_out = nn.Linear(mid_size, output_size) 
    self.bn = nn.BatchNorm1d(mid_size)

  def forward(self, x):
    x = F.relu(self.fc(x))
    for _ in range(self.mid_layers):
      x = F.relu(self.bn(self.fc_mid(x)))
    x = F.relu(self.fc_out(x))

    return x
from torch import optim

def calculate_loss_and_accuracy(model, criterion, loader, device):
  model.eval()
  loss = 0.0
  total = 0
  correct = 0
  with torch.no_grad():
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      loss += criterion(outputs, labels).item()
      pred = torch.argmax(outputs, dim=-1)
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return loss / len(loader), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
  # GPUに送る
  model.to(device)

  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

  # スケジューラの設定
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

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
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train, device)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid, device)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # チェックポイントの保存
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

    # 終了時刻の記録
    e_time = time.time()

    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 

    # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
    if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
      break

    # スケジューラを1ステップ進める
    scheduler.step()

  return {'train': log_train, 'valid': log_valid}

# datasetの作成
dataset_train = CreateDataset(X_train, y_train)
dataset_valid = CreateDataset(X_valid, y_valid)

# モデルの定義
model = MLPNet(300, 200, 4, 1)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# デバイスの指定
device = torch.device('cuda')

# モデルの学習
log = train_model(dataset_train, dataset_valid, 64, model, criterion, optimizer, 1000, device)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(np.array(log['train']).T[0], label='train')
ax[0].plot(np.array(log['valid']).T[0], label='valid')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[1].plot(np.array(log['train']).T[1], label='train')
ax[1].plot(np.array(log['valid']).T[1], label='valid')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()
plt.show()

def calculate_accuracy(model, loader, device):
  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      pred = torch.argmax(outputs, dim=-1)
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return correct / total

# 正解率の確認
acc_train = calculate_accuracy(model, dataloader_train, device)
acc_test = calculate_accuracy(model, dataloader_test, device)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')
