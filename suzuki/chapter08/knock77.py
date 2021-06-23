#knock77
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

model = LogisticRegression() #モデルの定義
ds = TensorDataset(X_train, y_train) #Datasetの定義
loss_fn = torch.nn.CrossEntropyLoss() #損失関数の定義

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
    print('batch size: {: <5}\ttime: {}'.format(bs, time.time() - start))


'''
*GPUで実行

batch size: 1    	time: 2.683497428894043
batch size: 2    	time: 1.529651403427124
batch size: 4    	time: 0.7751374244689941
batch size: 8    	time: 0.42537474632263184
batch size: 16   	time: 0.23553013801574707
batch size: 32   	time: 0.14814257621765137
batch size: 64   	time: 0.11170506477355957
batch size: 128  	time: 0.07869148254394531
batch size: 256  	time: 0.0672304630279541
batch size: 512  	time: 0.060502052307128906
batch size: 1024 	time: 0.056421756744384766
batch size: 2048 	time: 0.055207252502441406
batch size: 4096 	time: 0.055329084396362305
batch size: 8192 	time: 0.06167340278625488
batch size: 16384	time: 0.06356930732727051

'''