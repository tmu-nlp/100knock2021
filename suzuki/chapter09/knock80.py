import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_table('train.txt', header=None, sep='\t', names = ['category', 'title'])
valid = pd.read_table('valid.txt', header=None, sep='\t', names = ['category', 'title'])
test = pd.read_table('test.txt', header=None, sep='\t', names = ['category', 'title'])

vectorizer = CountVectorizer(min_df=2)
train_title = train['title'].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0) #各単語の出現頻度
idx = np.argsort(sm)[::-1] #単語を出現頻度でソート
words = np.array(vectorizer.get_feature_names())[idx] #出現頻度でソートされた単語ラベルの行列

print(len(words)) # 7612

d = dict()

for i in range(len(words)):
    d[words[i]] = i + 1

def get_id(sentence):
    r = []
    for word in sentence:
        r.append(d.get(word,0))
    return r

def df2id(df):
    ids = []
    for i in df.str.lower():
        ids.append(get_id(i.split()))
    return ids

X_train = df2id(train['title'])
X_valid = df2id(valid['title'])
X_test = df2id(test['title'])

