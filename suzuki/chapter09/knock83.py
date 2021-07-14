import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
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

train = pd.read_csv('train.txt', header=None, sep='\t') 
valid = pd.read_csv('valid.txt', header=None, sep='\t') 
test = pd.read_csv('test.txt', header=None, sep='\t') 

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
device = torch.device('cuda')
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
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

epoch: 0, Loss/train: 1.2732980251312256, Acc/train: 0.38571695994009736
epoch: 0, Loss/valid: 1.2752867937088013, Acc/valid: 0.375

epoch: 1, Loss/train: 1.2142391204833984, Acc/train: 0.4435604642456009
epoch: 1, Loss/valid: 1.2260754108428955, Acc/valid: 0.43862275449101795

epoch: 2, Loss/train: 1.1873633861541748, Acc/train: 0.4626544365406215
epoch: 2, Loss/valid: 1.2068017721176147, Acc/valid: 0.45434131736526945

epoch: 3, Loss/train: 1.1722491979599, Acc/train: 0.47163983526769
epoch: 3, Loss/valid: 1.1966992616653442, Acc/valid: 0.46781437125748504

epoch: 4, Loss/train: 1.1624455451965332, Acc/train: 0.4780044926993635
epoch: 4, Loss/valid: 1.1907389163970947, Acc/valid: 0.47155688622754494

epoch: 5, Loss/train: 1.155352234840393, Acc/train: 0.48202920254586296
epoch: 5, Loss/valid: 1.187362551689148, Acc/valid: 0.47904191616766467

epoch: 6, Loss/train: 1.1494883298873901, Acc/train: 0.4862411081991763
epoch: 6, Loss/valid: 1.1838805675506592, Acc/valid: 0.47679640718562877

epoch: 7, Loss/train: 1.1442948579788208, Acc/train: 0.4903594159490827
epoch: 7, Loss/valid: 1.180777907371521, Acc/valid: 0.4812874251497006

epoch: 8, Loss/train: 1.1396454572677612, Acc/train: 0.49438412579558216
epoch: 8, Loss/valid: 1.1781303882598877, Acc/valid: 0.48353293413173654

epoch: 9, Loss/train: 1.1351569890975952, Acc/train: 0.4970048670909772
epoch: 9, Loss/valid: 1.1750839948654175, Acc/valid: 0.48502994011976047


'''