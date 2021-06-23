from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import time


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1) #各単語ベクトルの要素の最大値
    label = label.data.numpy() #ラベルをpredと比べられる形にしてる?
    return (pred == label).mean() #正解率


X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')

list_batch_time = []

#cpu
model = LogisticRegression() #モデルの定義
device = torch.device('cpu')
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
loss_fn = torch.nn.CrossEntropyLoss()
batchs = [2**i for i in range(15)]

for bs in batchs:
    loader = DataLoader(ds, batch_size=bs, shuffle=True) # DataLoaderを作成
    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1) #オプティマイザの定義
    for epoch in range(1):
        start = time.time()
        for xx, yy in loader:
            optimizer.zero_grad() #勾配を0で初期化
            y_pred = model(xx) #categoryの予測
            loss = loss_fn(y_pred, yy) #損失率計算
            loss.backward() #誤差逆伝播
            optimizer.step() #重み更新
    list_batch_time.append([bs, time.time() - start])

#gpu
model = LogisticRegression() #モデルの定義
device = torch.device('cuda')
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
loss_fn = torch.nn.CrossEntropyLoss()
batchs = [2**i for i in range(15)]

for i, bs in enumerate(batchs):
    loader = DataLoader(ds, batch_size=bs, shuffle=True) # DataLoaderを作成
    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1) #オプティマイザの定義
    for epoch in range(1):
        start = time.time()
        for xx, yy in loader:
            optimizer.zero_grad() #勾配を0で初期化
            y_pred = model(xx) #categoryの予測
            loss = loss_fn(y_pred, yy) #損失率計算
            loss.backward() #誤差逆伝播
            optimizer.step() #重み更新
    list_batch_time[i].append(time.time() - start)

for case in list_batch_time:
    print('batch size: {: <5}\ttime: {:.3f}, {:.3f}'.format(case[0], case[1], case[2]))

'''

time: <cpu>, <gpu>

batch size: 1    	time: 2.760, 6.456
batch size: 2    	time: 1.547, 3.193
batch size: 4    	time: 0.793, 1.621
batch size: 8    	time: 0.427, 0.854
batch size: 16   	time: 0.245, 0.455
batch size: 32   	time: 0.150, 0.249
batch size: 64   	time: 0.104, 0.150
batch size: 128  	time: 0.082, 0.095
batch size: 256  	time: 0.072, 0.072
batch size: 512  	time: 0.061, 0.061
batch size: 1024 	time: 0.056, 0.053
batch size: 2048 	time: 0.054, 0.050
batch size: 4096 	time: 0.133, 0.131
batch size: 8192 	time: 0.061, 0.054
batch size: 16384	time: 0.141, 0.125

'''