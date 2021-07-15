'''
79. 多層ニューラルネットワーク
'''""

import torch
from torch import nn
from torch.nn import functional as F
from torch import cuda
import time
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from knock70 import transform_w2v
from knock73 import NewsDataset


# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']),
    ['TITLE', 'CATEGORY']]

# データの分割
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                               stratify=valid_test['CATEGORY'])
# 特徴ベクトルの作成
X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

# Datasetを作成するには、X_train, y_trainを利用
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)


# MLPNetを新たに定義
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


def calculate_loss_accuracy(model, criterion, loader, device):
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
    # モデルをGPUに送る
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs,eta_min=1e-5, last_epoch=-1)

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
        loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # チェックポイントの保存
        res_dir = './data/'
        model_param_dic = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_param_dic, res_dir + f'checkpoint_knock79_{epoch + 1}.pth')

        # 終了時刻の記録
        e_time = time.time()
        timeused = e_time - s_time

        # ログを出力
        print(
            f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, timeused : {timeused :.4f}sec')
        if epoch > 2 and log_valid[epoch -3][0] <= log_valid[epoch -2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
            break
        scheduler.step()

    return {'train': log_train, 'valid': log_valid}

if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = MLPNet(300, 200, 4 ,1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    log = train_model(dataset_train, dataset_valid, 64, model, criterion, optimizer, 100, device)

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

    acc_train = calculate_accuracy(model, dataset_train, device)
    acc_valid = calculate_accuracy(model, dataset_valid, device)
    print(f'accuracy on train:{acc_train:.3f}')
    print(f'accuracy on valid:{acc_valid:.3f}')


