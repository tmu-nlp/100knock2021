# 86. 畳み込みニューラルネットワーク (CNN)Permalink
# ID番号で表現された単語列x=(x1,x2,…,xT)がある．
# ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
# 畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，単語列xからカテゴリyを予測するモデルを実装せよ．

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
    item["inputs"] = torch.tensor(inputs, dtype=torch.int64)
    item["labels"] = torch.tensor(self.y[index], dtype=torch.int64)
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

    # モデルを定義する
    model = CNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4, emb_weights=weights)

    # 先頭10件の予測値取得
    for i in range(10):
        X = dataset_train[i]['inputs']
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

# 結果
# tensor([[0.2140, 0.2781, 0.3236, 0.1844]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2533, 0.2409, 0.3094, 0.1964]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2529, 0.2518, 0.2849, 0.2103]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2235, 0.2751, 0.2960, 0.2054]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2176, 0.2541, 0.2979, 0.2304]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2372, 0.2532, 0.2850, 0.2246]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2170, 0.2570, 0.3048, 0.2212]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2597, 0.2310, 0.2819, 0.2273]], grad_fn=<SoftmaxBackward>)
# tensor([[0.1948, 0.2471, 0.3375, 0.2206]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2173, 0.2628, 0.3141, 0.2058]], grad_fn=<SoftmaxBackward>)