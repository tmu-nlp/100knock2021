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
dw = 600
dh = 100
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
        self.conv = torch.nn.Conv1d(dw,dh,3,padding=1) # in_channels:dw, out_channels: dh
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
        y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
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

def rnn():

  model = RNN()
  device = torch.device('cuda')
  model.to(device)
  ds = TensorDataset(X_train.to(device), y_train.to(device))
  loader = DataLoader(ds, batch_size=1024, shuffle=True)
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

  for epoch in range(30):
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

def cnn():

  model = CNN()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  ds = TensorDataset(X_train.to(device), y_train.to(device))
  loader = DataLoader(ds, batch_size=1024, shuffle=True)
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

  for epoch in range(20):
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


rnn()
#cnn()

'''

ハイパーパラメータとイテレーションを変更

dw: 300 -> 600
dh: 50  -> 100
iteration: 200 -> 30

---------------------------------------------------------------------------

RNN

knock85
epoch: 9, Loss/train: 0.906947135925293, Acc/train: 0.6617371770872332
epoch: 9, Loss/valid: 0.9940864443778992, Acc/valid: 0.6175149700598802

knock88
epoch: 29, Loss/train: 0.342529833316803, Acc/train: 0.8871209284912018
epoch: 29, Loss/valid: 0.6931785345077515, Acc/valid: 0.7529940119760479

---------------------------------------------------------------------------

CNN

knock87
epoch: 9, Loss/train: 1.137510895729065, Acc/train: 0.6379633096218644
epoch: 9, Loss/valid: 1.1690194606781006, Acc/valid: 0.5793413173652695

knock88
epoch: 29, Loss/train: 0.9538324475288391, Acc/train: 0.8072819168850618
epoch: 29, Loss/valid: 1.104966640472412, Acc/valid: 0.6377245508982036

'''