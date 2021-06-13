import gensim
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
        self.rnn1 = torch.nn.RNN(dw,dh,bidirectional=True,batch_first=True)
        self.rnn2 = torch.nn.RNN(2*dh,dh,bidirectional=True,batch_first=True)
        self.rnn3 = torch.nn.RNN(2*dh,dh,bidirectional=True,batch_first=True)
        self.linear = torch.nn.Linear(2*dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn1(x, h)
        y, h = self.rnn2(y, h)
        y, h = self.rnn3(y, h)
        y = y[:,-1,:]
        y = self.linear(y)        
        return y

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw,dh,3,padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh,4)
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
        y = self.softmax(y)
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

model = CNN()
w2v = gensim.models.KeyedVectors.load_word2vec_format('../chapter07/data/GoogleNews-vectors-negative300.bin', binary=True)
for k, v in d.items():
    if k in w2v.vocab:
        model.emb.weight[v] = torch.tensor(w2v[k], dtype=torch.float32)
model.emb.weight = torch.nn.Parameter(model.emb.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device))
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))

"""
epoch: 0
train loss: 1.3211034536361694, train acc: 0.47763010108573567
valid loss: 1.3172059059143066, valid acc: 0.5067365269461078
epoch: 1
train loss: 1.2770555019378662, train acc: 0.5018719580681392
valid loss: 1.2695993185043335, valid acc: 0.5261976047904192
epoch: 2
train loss: 1.2523977756500244, train acc: 0.5213403219767877
valid loss: 1.2428096532821655, valid acc: 0.5419161676646707
epoch: 3
train loss: 1.2359493970870972, train acc: 0.5365967802321228
valid loss: 1.224961757659912, valid acc: 0.5441616766467066
epoch: 4
train loss: 1.2245415449142456, train acc: 0.5372519655559715
valid loss: 1.2123441696166992, valid acc: 0.5516467065868264
epoch: 5
train loss: 1.2157905101776123, train acc: 0.5446461999251216
valid loss: 1.2035034894943237, valid acc: 0.5494011976047904
epoch: 6
train loss: 1.209283471107483, train acc: 0.5495132909022837
valid loss: 1.1969125270843506, valid acc: 0.5494011976047904
epoch: 7
train loss: 1.2039587497711182, train acc: 0.5524148259078997
valid loss: 1.1919525861740112, valid acc: 0.5546407185628742
epoch: 8
train loss: 1.199321985244751, train acc: 0.5550355672032946
valid loss: 1.187880277633667, valid acc: 0.5561377245508982
epoch: 9
train loss: 1.195036768913269, train acc: 0.5599026581804567
valid loss: 1.1839007139205933, valid acc: 0.5651197604790419
"""