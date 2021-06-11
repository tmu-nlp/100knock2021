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
train loss: 1.1409881114959717, train acc: 0.5100149756645451
valid loss: 1.1431350708007812, valid acc: 0.5194610778443114
epoch: 1
train loss: 1.10617995262146, train acc: 0.5472669412205167
valid loss: 1.105824589729309, valid acc: 0.5606287425149701
epoch: 2
train loss: 1.0679901838302612, train acc: 0.5828341445151629
valid loss: 1.0643812417984009, valid acc: 0.593562874251497
epoch: 3
train loss: 1.0125998258590698, train acc: 0.6242044178210409
valid loss: 1.0079457759857178, valid acc: 0.6279940119760479
epoch: 4
train loss: 0.9341964721679688, train acc: 0.6645451141894422
valid loss: 0.9293990731239319, valid acc: 0.6736526946107785
epoch: 5
train loss: 0.9117649793624878, train acc: 0.6813927368026956
valid loss: 0.9105279445648193, valid acc: 0.6788922155688623
epoch: 6
train loss: 1.270799994468689, train acc: 0.5045862972669413
valid loss: 1.2830432653427124, valid acc: 0.4940119760479042
epoch: 7
train loss: 1.1128519773483276, train acc: 0.5588730812429802
valid loss: 1.116919755935669, valid acc: 0.5651197604790419
epoch: 8
train loss: 0.9473080635070801, train acc: 0.6402096593036316
valid loss: 0.9838783144950867, valid acc: 0.6287425149700598
epoch: 9
train loss: 0.7121379971504211, train acc: 0.7460688880569075
valid loss: 0.7293424606323242, valid acc: 0.7387724550898204
"""