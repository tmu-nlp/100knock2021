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

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
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

epoch: 0, Loss/train: 1.2740687131881714, Acc/train: 0.4319543242231374
epoch: 0, Loss/valid: 1.2791593074798584, Acc/valid: 0.4154191616766467

epoch: 1, Loss/train: 1.2512781620025635, Acc/train: 0.4997192062897791
epoch: 1, Loss/valid: 1.257187843322754, Acc/valid: 0.49101796407185627

epoch: 2, Loss/train: 1.2340624332427979, Acc/train: 0.5449269936353426
epoch: 2, Loss/valid: 1.241622805595398, Acc/valid: 0.5284431137724551

epoch: 3, Loss/train: 1.2176815271377563, Acc/train: 0.551666042680644
epoch: 3, Loss/valid: 1.227879285812378, Acc/valid: 0.5374251497005988

epoch: 4, Loss/train: 1.200284481048584, Acc/train: 0.5819917633845002
epoch: 4, Loss/valid: 1.2125276327133179, Acc/valid: 0.5494011976047904

epoch: 5, Loss/train: 1.1848849058151245, Acc/train: 0.5945338824410333
epoch: 5, Loss/valid: 1.2002416849136353, Acc/valid: 0.5658682634730539

epoch: 6, Loss/train: 1.171339750289917, Acc/train: 0.6002433545488581
epoch: 6, Loss/valid: 1.1902590990066528, Acc/valid: 0.5673652694610778

epoch: 7, Loss/train: 1.1588513851165771, Acc/train: 0.6165293897416698
epoch: 7, Loss/valid: 1.1818572282791138, Acc/valid: 0.5688622754491018

epoch: 8, Loss/train: 1.1477599143981934, Acc/train: 0.6256083863721452
epoch: 8, Loss/valid: 1.1749956607818604, Acc/valid: 0.5785928143712575

epoch: 9, Loss/train: 1.137510895729065, Acc/train: 0.6379633096218644
epoch: 9, Loss/valid: 1.1690194606781006, Acc/valid: 0.5793413173652695


'''