import numpy as np
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

#データ読み込み
X_train = np.loadtxt('./data/X_train.txt', delimiter=',')
X_train = torch.tensor(X_train, dtype=torch.float32) 
y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.long)

#単層ニューラルネット
class NN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, output_size, bias=False) #全結合層の定義
        torch.nn.init.normal_(self.layer1.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, input):
        activation = torch.nn.Softmax(dim=-1)
        output = activation(self.layer1(input))

        return output

model = NN(300, 4)
#(X, y)の組を作成
data_train = TensorDataset(X_train, y_train)
creterion = torch.nn.CrossEntropyLoss()
#最適化アルゴリズムの定義
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#学習
batch_size = [2**x for x in range(10)]
for batch in batch_size:
    dataloader = DataLoader(data_train, batch_size=batch, shuffle=True)
    for epoch in range(1):
        #計測開始
        start = time.time()
        for X, y in dataloader:
            optimizer.zero_grad() #勾配を0で初期化
            loss = creterion(model(X), y)
            loss.backward()
            optimizer.step() #パラメータ更新
        #計測終了
        t = time.time() - start
        print(f'time={t} sec (B={batch})')

'''
time=2.4380249977111816 sec (B=1)
time=1.2871370315551758 sec (B=2)
time=0.6769168376922607 sec (B=4)
time=0.3697052001953125 sec (B=8)
time=0.22745490074157715 sec (B=16)
time=0.1421642303466797 sec (B=32)
time=0.09511113166809082 sec (B=64)
time=0.10609626770019531 sec (B=128)
time=0.07455706596374512 sec (B=256)
time=0.07244420051574707 sec (B=512)
'''