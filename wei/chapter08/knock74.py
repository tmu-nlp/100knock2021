# -*- coding: utf-8 -*-
"""Task Description
74．正解率の計測
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''学習した単層モデルとDataloaderを入力として、正解率を算出する関数を定義'''


def calculate_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return correct / total


'''Dataloaderを準備'''

import pandas as pd
from sklearn.model_selection import train_test_split

# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE',
                                                                                                             'CATEGORY']]

# データの分割
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                               stratify=valid_test['CATEGORY'])

from gensim.models import KeyedVectors
import string
import torch

# 学習済み単語ベクトルを読み込む
model = KeyedVectors.load_word2vec_format('./../chapter07/data/GoogleNews-vectors-negative300.bin.gz', binary=True)


def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    words = text.translate(table).split()  # 記号をスペースに置換後、スペースで分割してリスト化
    vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化

    return torch.tensor(sum(vec) / len(vec))  # 平均ベクトルをTensor型に変換して出力


X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)


# 学習用のTensor型の平均化ベクトルとラベルベクトルを変換

class NewsDataset(Dataset):
    def __init__(self, X, y):  # datasetの構成要素を指定
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]で返す値を指定
        return [self.X[idx], self.y[idx]]


# 　Datasetを作成するには、X_train, y_train等を利用
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

# Dataloaderの作成
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

""" 単層NNモデルを作成"""

# SGLNetという単層ニューラルネットワークを定義
from torch import nn


class SGLNet(nn.Module):
    # 　ネットのlayer sizeを定義
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    # 　forwardで入力データが順伝播時に通るレイヤーを順に配置しておく
    def forward(self, x):
        x = self.fc(x)
        return x


# モデルの定義
SigleNNmodel = SGLNet(300, 4)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(SigleNNmodel.parameters(), lr=1e-1)

# 学習
num_epochs = 10
for epoch in range(num_epochs):
    # 訓練モードに設定
    SigleNNmodel.train()
    loss_train = 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
        # 勾配をゼロで初期化
        optimizer.zero_grad()

        # 順伝播 + 誤差逆伝播 + 重み更新
        outputs = SigleNNmodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


acc_train = calculate_accuracy(SigleNNmodel, dataloader_train)
acc_test = calculate_accuracy(SigleNNmodel, dataloader_test)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')



'''
正解率（学習データ）：0.924
正解率（評価データ）：0.903
'''