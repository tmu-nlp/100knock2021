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
train loss: 1.2225672006607056, train acc: 0.45544739797828526
valid loss: 1.222993016242981, valid acc: 0.4498502994011976
epoch: 1
train loss: 1.1656160354614258, train acc: 0.47379258704605015
valid loss: 1.1717300415039062, valid acc: 0.469311377245509
epoch: 2
train loss: 1.1500812768936157, train acc: 0.4851179333582928
valid loss: 1.159711241722107, valid acc: 0.47604790419161674
epoch: 3
train loss: 1.1404590606689453, train acc: 0.4995320104829652
valid loss: 1.1510484218597412, valid acc: 0.49101796407185627
epoch: 4
train loss: 1.1303168535232544, train acc: 0.5095469861475103
valid loss: 1.1413739919662476, valid acc: 0.5074850299401198
epoch: 5
train loss: 1.1174498796463013, train acc: 0.5341632347435418
valid loss: 1.1283011436462402, valid acc: 0.5419161676646707
epoch: 6
train loss: 1.096192717552185, train acc: 0.5556907525271434
valid loss: 1.1088876724243164, valid acc: 0.5666167664670658
epoch: 7
train loss: 1.0492582321166992, train acc: 0.6099775365031823
valid loss: 1.0625213384628296, valid acc: 0.625
epoch: 8
train loss: 0.9842209815979004, train acc: 0.6333770123549233
valid loss: 1.00222647190094, valid acc: 0.6347305389221557
epoch: 9
train loss: 0.8301785588264465, train acc: 0.716491950580307
valid loss: 0.8449915647506714, valid acc: 0.7073353293413174
"""