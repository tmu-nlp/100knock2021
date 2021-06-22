# 70.単語ベクトルの和による特徴量Permalink
'''
import re
import spacy
import torch
import pickle
from gensim.models import keyedvectors

nlp = spacy.load('en_core_web_sm')
categories = ['b', 't', 'e', 'm']
categories_names = ['business', 'science and technology', 'entertainment', 'health']

model = keyedvectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter08/GoogleNews-vectors-negative300.bin', binary = Ture)

def tokenize(x):
    x = re.sub(r'\s+', ' ', x)
    x = nlp.make_doc(x)
    x = [d.text for d in x]
    return x

def read_feature_dataset(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_t = [categories.index(line[0]) for line in dataset]
    dataset_x = [tokenize(line[1]) for line in dataset]
    return dataset_x, dataset_t

train_x, train_t = read_feature_dataset('/users/kcnco/github/100knock2021/pan/chapter08/train.txt')
valid_x, valid_t = read_feature_dataset('/users/kcnco/github/100knock2021/pan/chapter08/valid.txt')
test_x, test_t = read_feature_dataset('/users/kcnco/github/100knock2021/pan/chapter08/test.txt')

def sent_to_vector(sent):
    lst = [torch.tensor(model[token]) for token in sent if token in model]
    return sum(lst)/len(lst)

def dataset_to_vector(dataset):
    return torch.stack([sent_to_vector(x) for x in dataset])

train_v = dataset_to_vector(train_x)
valid_v = dataset_to_vector(valid_x)
test_v = dataset_to_vector(test_x)

train_v[0]

train_t = torch.tensor(train_t).long()
valid_t = torch.tensor(valid_t).long()
test_t = torch.tensor(test_t).long()

with open('/users/kcnco/github/100knock2021/pan/chapter08/train.feature.pickle', 'wb') as f:
    pickle.dump(train_v, f)
with open('/users/kcnco/github/100knock2021/pan/chapter08/train.label.pickle', 'wb') as f:
    pickle.dump(train_t, f)
with open('/users/kcnco/github/100knock2021/pan/chapter08/valid.feature.pickle', 'wb') as f:
    pickle.dump(valid_v, f)
with open('/users/kcnco/github/100knock2021/pan/chapter08/valid.label.pickle', 'wb') as f:
    pickle.dump(valid_t, f)
with open('/users/kcnco/github/100knock2021/pan/chapter08/test.feature.pickle', 'wb') as f:
    pickle.dump(test_v, f)
with open('/users/kcnco/github/100knock2021/pan/chapter08/test.label.pickle', 'wb') as f:
    pickle.dump(test_t, f)
'''
# 70. 単語ベクトルの和による特徴量
# 問題50で構築した学習データ，検証データ，評価データを行列・ベクトルに変換したい．
# 例えば，学習データについて，すべての事例xiの特徴ベクトルxiを並べた行列Xと，正解ラベルを並べた行列（ベクトル）Yを作成したい．
'''
from gensim.models import KeyedVectors
from scipy.special import logsumexp
import pandas as pd
import numpy as np
import joblib

def culcSwem(row):
    global model

    # 単語ベクトルに変換する
    swem = []
    for w in row['TITLE'].split():
        #if w in model.vocab:
        if w in model.vocab:
            swem.append(model[w])
        else:
            swem.append(np.zeros(shape=(model.vector_size,)))

    # 平均に変換する
    swem = np.mean(np.array(swem), axis = 0)

    return swem

if __name__ == '__main__':
    # データを読み込む
    X_train = pd.read_table('/users/kcnco/github/100knock2021/pan/chapter08/train.txt', header = None)
    X_valid = pd.read_table('/users/kcnco/github/100knock2021/pan/chapter08/valid.txt', header = None)
    X_test = pd.read_table('/users/kcnco/github/100knock2021/pan/chapter08/test.txt', header = None)
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter08/GoogleNews-vectors-negative300.bin', binary = True)

    # カラム名を設定する
    use_cols = ['TITLE', 'CATEGORY']
    X_train.columns = use_cols
    X_valid.columns = use_cols
    X_test.columns = use_cols

    # train、valid、testをまとめる
    data = pd.concat([X_train, X_valid, X_test]).reset_index(drop = True)

    # 特徴ベクトルを作成する
    swemVec = data.apply(culcSwem, axis = 1)

    # それぞれのカテゴリを自然数に変換する
    y_data = data['CATEGORY'].map({'b': 0, 'e': 1, 't': 2, 'm': 3})

    # それぞれの事例数を得る
    n_train = len(X_train)
    n_valid = len(X_valid)
    n_test = len(X_test)

    # train、valid、testに分割する
    X_train = np.array(list(swemVec.values)[:n_train])
    X_valid = np.array(list(swemVec.values)[n_train:n_train + n_valid])
    X_test = np.array(list(swemVec.values)[n_train + n_valid:])
    y_train = y_data.values[:n_train]
    y_valid = y_data.values[n_train:n_train + n_valid]
    y_test = y_data.values[n_train + n_valid:]

    # データを保存する
    joblib.dump(X_train, 'X_train.joblib')
    joblib.dump(X_valid, 'X_valid.joblib')
    joblib.dump(X_test, 'X_test.joblib')
    joblib.dump(y_train, 'y_train.joblib')
    joblib.dump(y_valid, 'y_valid.joblib')
    joblib.dump(y_test, 'y_test.joblib')
    '''

import pandas as pd
import gensim
import numpy as np

train = pd.read_csv('train.txt',sep='\t',header=None)
valid = pd.read_csv('valid.txt',sep='\t',header=None)
test = pd.read_csv('test.txt',sep='\t',header=None)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

d = {'b':0, 't':1, 'e':2, 'm':3}

y_train = train.iloc[:,0].replace(d)
y_train.to_csv('y_train.txt',header=False, index=False)
y_valid = valid.iloc[:,0].replace(d)
y_valid.to_csv('y_valid.txt',header=False, index=False)
y_test = test.iloc[:,0].replace(d)
y_test.to_csv('y_test.txt',header=False, index=False)

def write_X(file_name, df):
    with open(file_name,'w') as f:
        for text in df.iloc[:,1]:
            vectors = []
            for word in text.split():
                if word in model.vocab:
                    vectors.append(model[word])
            if (len(vectors)==0):
                vector = np.zeros(300)
            else:
                vectors = np.array(vectors)
                vector = vectors.mean(axis=0)
            vector = vector.astype(np.str).tolist()
            output = ' '.join(vector)+'\n'
            f.write(output)

write_X('X_train.txt', train)
write_X('X_valid.txt', valid)
write_X('X_test.txt', test)
