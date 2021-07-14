# 85. 双方向RNN・多層化
# 順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．

from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
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

def calculate_loss_and_accuracy(model, dataset, criterion=None):
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss = 0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in dataloader:
      inputs = data['inputs']
      labels = data['labels']
      outputs = model(inputs)
      if criterion != None:
        loss += criterion(outputs, labels).item()
      pred = torch.argmax(outputs, dim=-1)
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None):
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
  # スケジューラの設定
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

  for epoch in range(num_epochs):
    model.train()
    for data in dataloader_train:
      optimizer.zero_grad()
      inputs = data['inputs']
      labels = data['labels']
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    model.eval()
    # 損失と正解率を求める
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, criterion=criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, criterion=criterion)
    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train}, accuracy_train: {acc_train}, loss_valid: {loss_valid}, accuracy_valid: {acc_valid}')
    scheduler.step()

class Padsequence():
  # Dataloaderからミニバッチを取り出すごとに最大系列長でパディング
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
    labels = torch.LongTensor([x['labels'] for x in sorted_batch])
    return {'inputs': sequences_padded, 'labels': labels}

class CreateDataset(Dataset):
  # テキストとラベルを受け取り、テキストを指定したtokenizerでID化する
  def __init__(self, X, y, tokenizer, word2id):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer(text=text, word2id=word2id)
    return {'inputs': torch.tensor(inputs, dtype=torch.int64), 'labels': torch.tensor(self.y[index], dtype=torch.int64)}

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size=300, padding_idx=0, output_size=1, hidden_size=50, num_layers=1, bidirectional=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_directions = bidirectional + 1 # 単方向：1、双方向：2
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
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

    # パラメータを設定する
    VOCAB_SIZE = len(set(word2id.values())) + 1 # 辞書のID数 + パディングID
    PADDING_IDX = len(set(word2id.values()))
    BATCH_SIZE = 32
    NUM_EPOCHS = 30

    # モデルを定義する
    model = RNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4, bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

    # 正解率を算出する
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train acc: {acc_train}')
    print(f'test acc: {acc_test}')

# 結果
# epoch: 1, loss_train: 1.260977764802425, accuracy_train: 0.4269115442278861, loss_valid: 1.2591978961872732, accuracy_valid: 0.41304347826086957
# epoch: 2, loss_train: 1.2754669475956992, accuracy_train: 0.42325712143928035, loss_valid: 1.283132184469986, accuracy_valid: 0.4107946026986507
# epoch: 3, loss_train: 1.2924985858001585, accuracy_train: 0.4183845577211394, loss_valid: 1.3139424634346182, accuracy_valid: 0.41304347826086957
# epoch: 4, loss_train: 1.3000809334447494, accuracy_train: 0.42110194902548725, loss_valid: 1.3316187015783483, accuracy_valid: 0.4122938530734633
# epoch: 5, loss_train: 1.2989233221699563, accuracy_train: 0.42391304347826086, loss_valid: 1.335824945285313, accuracy_valid: 0.411544227886057
# epoch: 6, loss_train: 1.2688919754302328, accuracy_train: 0.44068590704647675, loss_valid: 1.303781578744548, accuracy_valid: 0.41829085457271364
# epoch: 7, loss_train: 1.2561380204198689, accuracy_train: 0.44818215892053975, loss_valid: 1.2865162402436114, accuracy_valid: 0.4287856071964018
# epoch: 8, loss_train: 1.2356623562954847, accuracy_train: 0.4687968515742129, loss_valid: 1.2658390484783781, accuracy_valid: 0.4482758620689655
# epoch: 9, loss_train: 1.224278981479044, accuracy_train: 0.47797976011994003, loss_valid: 1.251022144995708, accuracy_valid: 0.45577211394302847
# epoch: 10, loss_train: 1.2075128536489883, accuracy_train: 0.4906296851574213, loss_valid: 1.230101457343198, accuracy_valid: 0.46326836581709147
# epoch: 11, loss_train: 1.2008887985595564, accuracy_train: 0.496907796101949, loss_valid: 1.226797504682859, accuracy_valid: 0.46026986506746626
# epoch: 12, loss_train: 1.2015538284125022, accuracy_train: 0.49896926536731634, loss_valid: 1.2374474551120977, accuracy_valid: 0.4580209895052474
# epoch: 13, loss_train: 1.1699537710961083, accuracy_train: 0.5179910044977512, loss_valid: 1.1951375452206767, accuracy_valid: 0.48875562218890556
# epoch: 14, loss_train: 1.1594748900759255, accuracy_train: 0.525112443778111, loss_valid: 1.1865247875630767, accuracy_valid: 0.4955022488755622
# epoch: 15, loss_train: 1.1516758884216702, accuracy_train: 0.5323275862068966, loss_valid: 1.189575190040125, accuracy_valid: 0.4992503748125937
# epoch: 16, loss_train: 1.1294369024868267, accuracy_train: 0.5436656671664168, loss_valid: 1.1591713497976492, accuracy_valid: 0.5082458770614693
# epoch: 17, loss_train: 1.1140414433053945, accuracy_train: 0.5539730134932533, loss_valid: 1.1432100374815704, accuracy_valid: 0.5164917541229386
# epoch: 18, loss_train: 1.105638361465454, accuracy_train: 0.5608133433283359, loss_valid: 1.13699289591148, accuracy_valid: 0.5187406296851574
# epoch: 19, loss_train: 1.108009765904838, accuracy_train: 0.5588455772113943, loss_valid: 1.142982594735887, accuracy_valid: 0.5149925037481259
# epoch: 20, loss_train: 1.0948410891066271, accuracy_train: 0.5688718140929535, loss_valid: 1.1315126350481828, accuracy_valid: 0.5299850074962519
# epoch: 21, loss_train: 1.0871665913330997, accuracy_train: 0.5737443778110944, loss_valid: 1.1229331390052542, accuracy_valid: 0.5322338830584707
# epoch: 22, loss_train: 1.0793727316895034, accuracy_train: 0.5805847076461769, loss_valid: 1.115182637464607, accuracy_valid: 0.5389805097451275
# epoch: 23, loss_train: 1.0675788809220004, accuracy_train: 0.5875187406296851, loss_valid: 1.0972405206049043, accuracy_valid: 0.5524737631184408
# epoch: 24, loss_train: 1.073263944441528, accuracy_train: 0.5843328335832084, loss_valid: 1.108073742794937, accuracy_valid: 0.5434782608695652
# epoch: 25, loss_train: 1.0722036431145587, accuracy_train: 0.5852698650674663, loss_valid: 1.1109118382523264, accuracy_valid: 0.5367316341829086
# epoch: 26, loss_train: 1.0683605755955774, accuracy_train: 0.5882683658170914, loss_valid: 1.1057760789431912, accuracy_valid: 0.5449775112443778
# epoch: 27, loss_train: 1.0682716550817923, accuracy_train: 0.5887368815592204, loss_valid: 1.105508899771381, accuracy_valid: 0.5442278860569715
# epoch: 28, loss_train: 1.0666310514352735, accuracy_train: 0.5899550224887556, loss_valid: 1.1047349016571688, accuracy_valid: 0.5494752623688156
# epoch: 29, loss_train: 1.0664820025236565, accuracy_train: 0.5901424287856072, loss_valid: 1.1049608956391308, accuracy_valid: 0.547976011994003
# epoch: 30, loss_train: 1.0668559644561166, accuracy_train: 0.5896739130434783, loss_valid: 1.1056253032080237, accuracy_valid: 0.5464767616191905
# train acc: 0.5896739130434783
# test acc: 0.5899550224887556