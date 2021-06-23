# -*- coding: utf-8 -*-
"""Task Description   
70. 単語ベクトルの和による特徴量

学習データについて，すべての事例xi(r.v.)の特徴ベクトルxiを並べた行列Xと，正解ラベルを並べた行列（ベクトル）Yを作成する.
ラベルの種類数をL(L=4)で表す.i番目の事例はTi個の（記事見出しの）単語列(wi,1,wi,2,…,wi,Ti)から構成される。即ち、i番目の事例の記事見出しを，
その見出しに含まれる単語のベクトルの平均で表現したものがxiである学習データ、検証データ、評価データそれぞれの特徴量行列及びラベルベクトルを作成し、ファイルに保存
"""""

import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import string
import torch
import numpy as np
# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE',
                                                                                                             'CATEGORY']]

# データの分割(stratify->按原数据中各比例分配给train和test，以使各类数据比例和原数据集一致)
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                               stratify=valid_test['CATEGORY'])

# 学習済み単語ベクトルを読み込む,use gzipped/bz2 as input, and no need to unzip.
# when loading .bin file(二进制文件)，need make binary=True
model = KeyedVectors.load_word2vec_format('./../chapter07/data/GoogleNews-vectors-negative300.bin.gz', binary=True)


def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    words = text.translate(table).split()  # 記号をスペースに置換後、スペースで分割してリスト化
    vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化

    return torch.tensor(sum(vec) / len(vec))  # 平均ベクトルをTensor型に変換して出力


# 特徴ベクトルの作成
X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])

print(X_train.size())
print(X_train)

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

print(y_train.size())
print(y_train)



# 保存
data_dir = './data/'
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)
np.savetxt(data_dir + 'X_train.txt', X_train)
np.savetxt(data_dir + 'X_valid.txt', X_valid)
np.savetxt(data_dir + 'X_test.txt', X_test)
np.savetxt(data_dir + 'y_train.txt', y_train)
np.savetxt(data_dir + 'y_valid.txt', y_valid)
np.savetxt(data_dir + 'y_test.txt', y_test)



