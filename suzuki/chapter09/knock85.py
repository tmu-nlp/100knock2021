import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader

train = pd.read_csv('train.txt', header=None, sep='\t') 
valid = pd.read_csv('valid.txt', header=None, sep='\t') 
test = pd.read_csv('test.txt', header=None, sep='\t') 

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
    pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
    label = label.data.to('cpu').numpy()
    return (pred == label).mean()

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')
y_test = torch.load('y_test.pt')

model = RNN()
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
        print ('epoch: {}, Loss/train: {}, Acc/train: {}'.format(epoch, loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print ('epoch: {}, Loss/valid: {}, Acc/valid: {}'.format(epoch, loss.item(), accuracy(y_pred,y_valid)))
        print('')

'''

epoch: 0, Loss/train: 1.148280143737793, Acc/train: 0.5077686259827779
epoch: 0, Loss/valid: 1.1589361429214478, Acc/valid: 0.5022455089820359

epoch: 1, Loss/train: 1.1025662422180176, Acc/train: 0.5436166229876451
epoch: 1, Loss/valid: 1.126943826675415, Acc/valid: 0.5232035928143712

epoch: 2, Loss/train: 1.0730689764022827, Acc/train: 0.5611194309247473
epoch: 2, Loss/valid: 1.1074212789535522, Acc/valid: 0.5336826347305389

epoch: 3, Loss/train: 1.0464829206466675, Acc/train: 0.5842381130662673
epoch: 3, Loss/valid: 1.0865025520324707, Acc/valid: 0.5523952095808383

epoch: 4, Loss/train: 1.0204297304153442, Acc/train: 0.6009921377761138
epoch: 4, Loss/valid: 1.0705353021621704, Acc/valid: 0.5673652694610778

epoch: 5, Loss/train: 0.9957584738731384, Acc/train: 0.6151254211905653
epoch: 5, Loss/valid: 1.052182912826538, Acc/valid: 0.5763473053892215

epoch: 6, Loss/train: 0.9724667072296143, Acc/train: 0.6254211905653313
epoch: 6, Loss/valid: 1.03856360912323, Acc/valid: 0.5830838323353293

epoch: 7, Loss/train: 0.9523876309394836, Acc/train: 0.6349681767128417
epoch: 7, Loss/valid: 1.028970718383789, Acc/valid: 0.5875748502994012

epoch: 8, Loss/train: 0.9264886975288391, Acc/train: 0.6505054286783976
epoch: 8, Loss/valid: 1.0070784091949463, Acc/valid: 0.6040419161676647

epoch: 9, Loss/train: 0.906947135925293, Acc/train: 0.6617371770872332
epoch: 9, Loss/valid: 0.9940864443778992, Acc/valid: 0.6175149700598802


'''