import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

base = '../chapter06/'
train = pd.read_csv(base + 'train.txt', header=None, sep='\t')
valid = pd.read_csv(base + 'valid.txt', header=None, sep='\t')
test = pd.read_csv(base + 'test.txt', header=None, sep='\t')

vectorizer = CountVectorizer(min_df=2)  # TFを計算。ただし、出現頻度が2回以上の単語だけを登録
train_title = train.iloc[:, 0].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()  # title corpusを入力とし、TF array(スパース行列)を得る

sm = cnt.sum(axis=0)  # 列ごとに累加して、.get_feature_names()の単語ごとに、各docに出現頻度を数える
idx = np.argsort(sm)[::-1]  # 出現頻度の降順で、対応するindexを返す(.argsort返回数组值从小到大的对应索引值)
words = np.array(vectorizer.get_feature_names())[idx]  # ['w1',...,'wn'][index] indexで単語を索引し返す。最も出現した単語が先頭に

dw = 300
dh = 50
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(len(words)+1,dw)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        y = self.softmax(y)
        return y