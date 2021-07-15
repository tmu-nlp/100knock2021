""'''
82. 確率的勾配降下法による学習
'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from knock80_ import df2id
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

base1 = '../chapter06/'

train = pd.read_csv(base1 + 'train.txt', header=None, sep='\t')
valid = pd.read_csv(base1 + 'valid.txt', header=None, sep='\t')
test = pd.read_csv(base1 + 'test.txt', header=None, sep='\t')

vectorizer = CountVectorizer(min_df=2)                 # TFを計算。ただし、出現頻度が2回以上の単語だけを登録
train_title = train.iloc[:, 0].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()  # title corpusを入力とし、TF array(スパース行列)を得る

sm = cnt.sum(axis=0)                  # 列ごとに累加して、.get_feature_names()の単語ごとに、各docに出現頻度を数える
idx = np.argsort(sm)[::-1]            # 出現頻度の降順で、対応するindexを返す(.argsort返回数组值从小到大的对应索引值)
words = np.array(vectorizer.get_feature_names())[idx]          # ['w1',...,'wn'][index] indexで単語を索引し返す。最も出現した単語が先頭に

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab, dw, padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw, dh, batch_first=True)
        self.linear = torch.nn.Linear(dh, 4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:, -1, :]
        y = self.linear(y)
        return y


def list2tensor(data, max_len):
    new = []
    for d in data:                  # data: [行=文書毎のID番号のリスト]
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
X_train = list2tensor(X_train, max_len)
X_valid = list2tensor(X_valid, max_len)
X_test = list2tensor(X_test, max_len)

y_train = np.loadtxt('y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = RNN()
ds = TensorDataset(X_train, y_train)
# Dataloaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

if __name__ == '__main__':

    for epoch in range(10):
        for xx, yy in loader:
            y_pred = model(xx)
            loss = loss_func(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = model(X_train)
            loss = loss_func(y_pred, y_train)
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy(y_pred, y_train), epoch)
            print(f'epoch:{epoch+1}')
            print(f'Accuracy on train:{accuracy(y_pred, y_train)}')


            y_pred = model(X_valid)
            loss = loss_func(y_pred, y_valid)
            writer.add_scalar('Loss/valid', loss, epoch)
            writer.add_scalar('Accuracy/valid', accuracy(y_pred, y_valid), epoch)
            print(f'Accuracy on valid:{accuracy(y_pred, y_valid)}')



'''
lr = 1e-1
epoch:10
Accuracy on train:0.4335454885810558
Accuracy on valid:0.42664670658682635

lr = 1e-3
epoch:10
Accuracy on train:0.8346125046798951
Accuracy on valid:0.7537425149700598

'''



