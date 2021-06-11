import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader

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

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:]
        y = self.linear(y)
        return y

def list2tensor(data, max_len):
    new = []
    for d in data:
        if len(d) > max_len:
            d = d[:max_len]
        else:
            d += [PAD] * (max_len - len(d))
        new.append(d)
    return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)
    label = label.data.numpy()
    return (pred == label).mean()

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt('../chapter08/data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('../chapter08/data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('../chapter08/data/y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = RNN()
ds = TensorDataset(X_train, y_train)
loader = DataLoader(ds, batch_size=1, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(2):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid)
        loss = loss_fn(y_pred, y_valid)
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))

"""
epoch: 0
train loss: 2.8293888568878174, train acc: 0.3728940471733433
valid loss: 2.9555394649505615, valid acc: 0.3712574850299401
epoch: 1
train loss: 3.406196355819702, train acc: 0.37776113815050544
valid loss: 3.607290029525757, valid acc: 0.35254491017964074
"""