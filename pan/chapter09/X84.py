# 84. 単語ベクトルの導入Permalink
# 事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．

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
from gensim.models import KeyedVectors
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


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn = None, device = None):
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size = 1, shuffle = False)
  # スケジューラの設定
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = 1e-5, last_epoch = -1)

  log_train = []
  log_valid = []
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
    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion = criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion = criterion)
    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train}, accuracy_train: {acc_train}, loss_valid: {loss_valid}, accuracy_valid: {acc_valid}') 
    scheduler.step() # スケジューラを1ステップ進める

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

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size = 300, padding_idx = 0, output_size = 1, hidden_size = 50, num_layers = 1, emb_weights = None, bidirectional = False):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_directions = bidirectional + 1
    if emb_weights != None:
      self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx = padding_idx)
    else:
      self.emb = nn.Embedding(vocab_size, emb_size, padding_idx = padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity = 'tanh', bidirectional = bidirectional, batch_first = True)
    self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()
    emb = self.emb(x)
    out, hidden = self.rnn(emb, hidden)
    out = self.fc(out[:, -1, :])
    return out

  def init_hidden(self):
    hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
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

    # 学習済み単語ベクトルの取得
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale = 0.4, size = (EMB_SIZE,))
    weights = torch.from_numpy(weights.astype((np.float32)))

    # パラメータを設定する
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    BATCH_SIZE = 32
    NUM_EPOCHS = 3

    # モデルを定義する
    model = RNN(vocab_size = VOCAB_SIZE, padding_idx = PADDING_IDX, output_size = 4, emb_weights = weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn = Padsequence(PADDING_IDX))

    # 正解率を算出する
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train acc: {acc_train}')
    print(f'test acc: {acc_test}')

# 結果
# epoch: 1, loss_train: 1.1698, accuracy_train: 0.4190, loss_valid: 1.1636, accuracy_valid: 0.4175
# epoch: 2, loss_train: 1.1665, accuracy_train: 0.4386, loss_valid: 1.1607, accuracy_valid: 0.4475
# epoch: 3, loss_train: 1.1665, accuracy_train: 0.4461, loss_valid: 1.1611, accuracy_valid: 0.4528
# train acc: 0.446
# test acc: 0.448