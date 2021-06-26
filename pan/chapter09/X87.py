# 87. 確率的勾配降下法によるCNNの学習
# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題86で構築したモデルを学習せよ．
# 訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import defaultdict
from torch import optim
from torch import nn
import pandas as pd
import numpy as np
import string
import torch
import time

def tokenizer(text, word2id, unk=0):
  ids = []
  for word in text.split():
      ids.append(word2id.get(word, unk))
  return ids

def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss = 0
  total = 0
  correct = 0
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
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device=None):
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
  # スケジューラの設定
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

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
    # 損失と正解率を求める
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion=criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion=criterion)
    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train}, accuracy_train: {acc_train}, loss_valid: {loss_valid}, accuracy_valid: {acc_valid}') 
    scheduler.step()

class Padsequence():
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
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
    inputs = self.tokenizer(text=text, word2id=word2id)
    item = {}
    item['inputs'] = torch.tensor(inputs, dtype=torch.int64)
    item['labels'] = torch.tensor(self.y[index], dtype=torch.int64)
    return item

class CNN(nn.Module):
  def __init__(self, vocab_size, emb_size=300, padding_idx=0, output_size=1, out_channels=100, kernel_heights=3, stride=1, padding=1, emb_weights=None):
    super().__init__()
    if emb_weights != None:
      self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
    else:
      self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
    self.drop = nn.Dropout(0.3)
    self.fc = nn.Linear(out_channels, output_size)

  def forward(self, x):
    emb = self.emb(x).unsqueeze(1)
    conv = self.conv(emb)
    act = F.relu(conv.squeeze(3))
    max_pool = F.max_pool1d(act, act.size()[2])
    out = self.fc(self.drop(max_pool.squeeze(2)))
    return out

if __name__ == '__main__':
    # データを読み込む
    train = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/train.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    valid = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/valid.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    test = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/test.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])

    # 単語の頻度を集計する
    d = defaultdict(int)
    for text in train['TITLE']:
        for word in text.split():
            d[word] += 1
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)

    # 単語ID辞書を作成する
    word2id = {}
    for i, (word, cnt) in enumerate(d):
        # 出現頻度が2回以上の単語を登録
        if cnt <= 1:
            continue
        word2id[word] = i + 1

    # ラベルを数値に変換する
    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
    y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
    y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

    # Datasetの作成
    dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, word2id)
    dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, word2id)
    dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, word2id)

    # 学習済みモデルを読み込む
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter09/GoogleNews-vectors-negative300.bin', binary=True)

    # 学習済み単語ベクトルを取得する
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    weights = torch.from_numpy(weights.astype((np.float32)))

    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    # モデルを定義する
    model = CNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4, emb_weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train acc: {acc_train}')
    print(f'test acc: {acc_test}')

# 結果
# epoch: 1, loss_train: 1.1512, accuracy_train: 0.5070, loss_valid: 1.1450, accuracy_valid: 0.5225
# epoch: 2, loss_train: 1.1161, accuracy_train: 0.5402, loss_valid: 1.1174, accuracy_valid: 0.5420
# epoch: 3, loss_train: 1.0932, accuracy_train: 0.5614, loss_valid: 1.1009, accuracy_valid: 0.5442
# epoch: 4, loss_train: 1.0786, accuracy_train: 0.5712, loss_valid: 1.0907, accuracy_valid: 0.5495
# epoch: 5, loss_train: 1.0679, accuracy_train: 0.5797, loss_valid: 1.0810, accuracy_valid: 0.5622
# epoch: 6, loss_train: 1.0605, accuracy_train: 0.5833, loss_valid: 1.0759, accuracy_valid: 0.5705
# epoch: 7, loss_train: 1.0554, accuracy_train: 0.5882, loss_valid: 1.0714, accuracy_valid: 0.5727
# epoch: 8, loss_train: 1.0524, accuracy_train: 0.5884, loss_valid: 1.0692, accuracy_valid: 0.5720
# epoch: 9, loss_train: 1.0510, accuracy_train: 0.5897, loss_valid: 1.0680, accuracy_valid: 0.5735
# epoch: 10, loss_train: 1.0506, accuracy_train: 0.5900, loss_valid: 1.0677, accuracy_valid: 0.5735
# train acc: 0.590
# test acc: 0.581
