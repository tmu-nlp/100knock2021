# 89. 事前学習済み言語モデルからの転移学習
# 事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import defaultdict
from torch import optim
from torch import cuda
from torch import nn
import transformers
import pandas as pd
import numpy as np
import string
import torch
import time

def calculate_loss_and_accuracy(model, criterion, loader, device=None):
  model.eval()
  loss = 0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)
      outputs = model.forward(ids, mask)
      # 損失を加算する
      loss += criterion(outputs, labels).item()
      # 正解数を数える
      pred = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy()
      total += len(labels)
      correct += (pred == labels).sum().item()
  return loss / len(loader), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
  for epoch in range(num_epochs):
    model.train()
    for data in dataloader_train:
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)
      optimizer.zero_grad()
      outputs = model.forward(ids, mask)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    # 損失と正解率を求める
    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
    print(f'epoch: {epoch + 1}, loss_train: {loss_train}, accuracy_train: {acc_train}, loss_valid: {loss_valid}, accuracy_valid: {acc_valid}')

def calculate_accuracy(model, dataset, device=None):
  # Dataloaderの作成
  loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)
      outputs = model.forward(ids, mask)
      pred = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy()
      total += len(labels)
      correct += (pred == labels).sum().item()
  return correct / total

# BERT分類モデルの定義
class BERTClass(torch.nn.Module):
  def __init__(self, drop_rate=0.4, output_size=1):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.drop = torch.nn.Dropout(drop_rate)
    self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

  def forward(self, ids, mask):
    _, out = self.bert(ids, attention_mask=mask)
    out = self.fc(self.drop(out))
    return out

class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      pad_to_max_length=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    return {
      'ids': torch.LongTensor(ids),
      'mask': torch.LongTensor(mask),
      'labels': torch.Tensor(self.y[index])
    }

if __name__ == '__main__':
    # データを読み込む
    df = pd.read_csv('/users/kcnco/github/100knock2021/pan/chapter09/newsCorpora_re.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # 正解ラベルをone-hot化する
    y_train = pd.get_dummies(train, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_valid = pd.get_dummies(valid, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_test = pd.get_dummies(test, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

    # Datasetを作成する
    max_len = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, max_len)
    dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, max_len)
    dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, max_len)

    # ラベルを数値に変換する
    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
    y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
    y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

    # パラメータを設定する
    BATCH_SIZE = 32
    NUM_EPOCHS = 1

    # モデルを定義する
    model = BERTClass(output_size=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

    # 正解率を求める
    print(f'train acc: {calculate_accuracy(model, dataset_train)}')
    print(f'valid acc: {calculate_accuracy(model, dataset_valid)}')
    print(f'test  acc: {calculate_accuracy(model, dataset_test)}')