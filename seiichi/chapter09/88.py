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
optimizer = torch.optim.Adam(model.parameters())

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
train loss: 1.23563814163208, train acc: 0.5384687383002621
valid loss: 1.2231026887893677, valid acc: 0.5538922155688623
epoch: 1
train loss: 1.1862462759017944, train acc: 0.5862036690378135
valid loss: 1.174551248550415, valid acc: 0.5875748502994012
epoch: 2
train loss: 1.1500271558761597, train acc: 0.6351553725196556
valid loss: 1.145823359489441, valid acc: 0.6197604790419161
epoch: 3
train loss: 1.107067584991455, train acc: 0.6825159116435792
valid loss: 1.1106376647949219, valid acc: 0.6729041916167665
epoch: 4
train loss: 1.0565885305404663, train acc: 0.7331523773867465
valid loss: 1.0656235218048096, valid acc: 0.7148203592814372
epoch: 5
train loss: 1.01140296459198, train acc: 0.7702171471359042
valid loss: 1.0239462852478027, valid acc: 0.750748502994012
epoch: 6
train loss: 0.979972243309021, train acc: 0.7858479970048671
valid loss: 0.9975193738937378, valid acc: 0.7679640718562875
epoch: 7
train loss: 0.9617096781730652, train acc: 0.79623736428304
valid loss: 0.983503520488739, valid acc: 0.7724550898203593
epoch: 8
train loss: 0.9500837922096252, train acc: 0.8018532384874579
valid loss: 0.9749095439910889, valid acc: 0.7769461077844312
epoch: 9
train loss: 0.9423613548278809, train acc: 0.8053163609135155
valid loss: 0.9696382284164429, valid acc: 0.7799401197604791
"""