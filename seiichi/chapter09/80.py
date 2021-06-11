import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('../chapter06/data/train.txt', header=None, sep='\t')
valid = pd.read_csv('../chapter06/data/valid.txt', header=None, sep='\t')
test = pd.read_csv('../chapter06/data/test.txt', header=None, sep='\t') 

vectorizer = CountVectorizer(min_df=2)
train_title = train.iloc[:,1].str.lower()

cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()

for i in range(len(words)):
  d[words[i]] = i+1

def get_id(sentence):
    r = []
    for word in sentence:
        r.append(d.get(word,0))
    return r

def df2id(df):
    ids = []
    for i in df.iloc[:,1].str.lower():
        ids.append(get_id(i.split()))
    return ids

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)