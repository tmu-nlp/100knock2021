path = '/content/drive/MyDrive/basis2021/nlp100/'
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from tqdm import tqdm
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import time

#データ読み込み
train_df = pd.read_table(path + 'train.txt')
valid_df = pd.read_table(path + 'valid.txt')
test_df = pd.read_table(path + 'test.txt')

#データ抜き出し
X_train = train_df['TITLE']
X_valid = valid_df['TITLE']
X_test = test_df['TITLE']

y_train = train_df['CATEGORY']
y_valid = valid_df['CATEGORY']
y_test = test_df['CATEGORY']

category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = y_train.map(lambda x: category_dict[x]).values
y_valid = y_valid.map(lambda x: category_dict[x]).values
y_test = y_test.map(lambda x: category_dict[x]).values

#単語の頻度を計算
word_dicts = defaultdict(lambda: 0)
for a_text in X_train:
    for word in a_text.split():
        word_dicts[word] += 1
word_dicts = sorted(word_dicts.items(), key = lambda x: x[1], reverse=True)

#頻度の多い単語からid付与
word2id = {}
for i, (word, freq) in enumerate(word_dicts):
    if freq <= 1:
        continue
    word2id[word] = i+1

#与えられた単語列に対してIDの列を返す関数
def tokenizer(text):
    ids = []
    for word in text.split():
        #2回未満の単語は辞書にないので0を返す
        ids.append(word2id.get(word, 0))
    return ids

if __name__ == "__main__":
    text = 'This sentence is rondom'
    print(text)
    tokenized_text = tokenizer(text)
    y = torch.tensor(tokenized_text)
    print(y)
    for key in list(word2id)[:20]:
        print(f'{key}: {word2id[key]}')
