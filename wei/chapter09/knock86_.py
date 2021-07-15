""'''
86．畳込みニューラルネット
CNNを用いて、単語列ｘからカテゴリｙを予測するモデルを実装'''


import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from knock80_ import df2id
from knock82_ import list2tensor
from knock83_ import accuracy_gpu
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


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
    d[words[i]] = i + 1


max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.emb = torch.nn.Embedding(n_vocab, dw, padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw, dh, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh, 4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        y = self.linear(x)
        #　y = self.softmax(y)
        return y


X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)
X_train = list2tensor(X_train, max_len)
X_valid = list2tensor(X_valid, max_len)
X_test = list2tensor(X_test, max_len)

y_train = np.loadtxt('y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)


model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


with torch.no_grad():
    y_pred = model(X_train.to(device))

    print(f'\naccuracy on train:{accuracy_gpu(y_pred, y_train)}')
    y_pred = model(X_valid.to(device))

    print(f'accuracy on valid:{accuracy_gpu(y_pred, y_valid)}')


'''
without training model
accuracy on train:0.24925121677274428
accuracy on valid:0.23652694610778444
'''



