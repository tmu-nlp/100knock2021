import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

base = '../chapter06/'
train = pd.read_csv(base + 'train.txt', header=None, sep='\t')
valid = pd.read_csv(base + 'valid.txt', header=None, sep='\t')
test = pd.read_csv(base + 'test.txt', header=None, sep='\t')

vectorizer = CountVectorizer(min_df=2)    #TFを計算。ただし、出現頻度が2回以上の単語だけを登録
train_title = train.iloc[:, 0].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()  #title corpusを入力とし、各文書を行に、.get_feature_names()を列に、TF array(スパース行列)を得る

sm = cnt.sum(axis=0)            # 列ごとに累加して、.get_feature_names()の単語ごとに、各docに出現頻度を数える
idx = np.argsort(sm)[::-1]      # 出現頻度の降順で、対応するindexを返す(.argsort返回数组值从小到大的对应索引值)
words = np.array(vectorizer.get_feature_names())[idx]   # ['w1',...,'wn'][index] indexで単語を索引し返す。最も出現した単語が先頭に
d = dict()
for i in range(len(words)):
    d[words[i]] = i + 1         # 最も頻出する単語に1という方法で、単語にID番号を付与し辞書を作成


def get_id(sentence):
    r = []
    for word in sentence:
        r.append(d.get(word, 0))
    return r                     # 単語に対応するID番号のリストを返す. ID番号の小さいほど、最も出現


def df2id(df):
    ids = []
    for i in df.iloc[:, 0].str.lower():
        ids.append(get_id(i.split()))   # 行=文書毎のID番号のリストを追加
    return ids

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

if __name__ == '__main__':
    for i in range(5):
        print(X_train[i])

