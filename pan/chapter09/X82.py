# 82. 確率的勾配降下法による学習
# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
# 訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．

import time
import torch
import string
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def tokenizer(text, word2id, unk = 0):
  ids = []
  for word in text.split():
      ids.append(word2id.get(word, unk))
  return ids

def calculate_loss_and_accuracy(model, dataset, device = None, criterion = None):
  dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
  loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for data in dataloader:
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = model(inputs)
      # 損失を加算する
      if criterion != None:
        loss += criterion(outputs, labels).item()
      # 正解数を数える
      pred = torch.argmax(outputs, dim=-1)
      correct += (pred == labels).sum().item()
      total += len(inputs)

  return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn = None, device = None):
  model.to(device)
  # dataloaderを作成する
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle = True, collate_fn = collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size = 1, shuffle = False)
  # スケジューラを設定する
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = 1e-5, last_epoch = -1)

  for epoch in range(num_epochs):
    model.train()
    for data in dataloader_train:
      optimizer.zero_grad()
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    model.eval()
    # 損失と正解率を算出する
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion = criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion = criterion)
    # ログを出力する
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}') 
    scheduler.step() # スケジューラを1ステップ進める

class Padsequence():
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first = True, padding_value = self.padding_idx)
    labels = torch.LongTensor([x['labels'] for x in sorted_batch])
    call = {}
    call['inputs'] = sequences_padded
    call['labels'] = labels
    return call

class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, word2id):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer(text = text, word2id = word2id)
    item = {}
    item['inputs'] = torch.tensor(inputs, dtype = torch.int64)
    item['labels'] = torch.tensor(self.y[index], dtype = torch.int64)
    return item

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size = 300, padding_idx = 0, output_size = 1, hidden_size = 50):
    super().__init__()
    self.hidden_size = hidden_size
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx = padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity = 'tanh', batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()
    emb = self.emb(x)
    out, hidden = self.rnn(emb, hidden)
    out = self.fc(out[:, -1, :])
    return out

  def init_hidden(self):
    hidden = torch.zeros(1, self.batch_size, self.hidden_size)
    return hidden

if __name__ == '__main__':
    train = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/train.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    valid = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/valid.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    test = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/test.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])

    # 単語の頻度を集計する
    d = defaultdict(int)
    for text in train['TITLE']:
        for word in text.split():
            d[word] += 1
    d = sorted(d.items(), key = lambda x:x[1], reverse = True)

    # 単語ID辞書を作成する
    word2id = {}
    for i, (word, cnt) in enumerate(d):
        # 出現頻度が2回以上の単語を登録する
        if cnt <= 1:
            continue
        word2id[word] = i + 1

    # ラベルを数値に変換する
    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
    y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
    y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

    # Datasetを作成する
    dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, word2id)
    dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, word2id)
    dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, word2id)

    # パラメータを設定する
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    BATCH_SIZE = 1
    NUM_EPOCHS = 3

    # モデルを定義する
    model = RNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

    # 正解率を表示する
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train acc: {acc_train}')
    print(f'test acc: {acc_test}')