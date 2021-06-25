import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

#データ読み込み
X_train = np.loadtxt('./data/X_train.txt', delimiter=',')
X_train = torch.tensor(X_train, dtype=torch.float32) 
y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.long)
X_valid = np.loadtxt('./data/X_valid.txt', delimiter=',')
X_valid = torch.tensor(X_valid, dtype=torch.float32) 
y_valid = np.loadtxt('./data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.long)

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
#DataLoaderの作成
dataloader = DataLoader(data_train, batch_size=1, shuffle=True)
creterion = torch.nn.CrossEntropyLoss()
#最適化アルゴリズムの定義
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def accuracy(probs, y):
    cnt = 0
    for i, prob in enumerate(probs):
        #tensorからndarrayに変換し、最大要素のindexを返す
        y_pred = np.argmax(prob.detach().numpy())
        if y_pred == y.detach().numpy()[i]:
            cnt += 1
    
    return cnt/len(y)

#学習
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(10):
    for X, y in dataloader:
        optimizer.zero_grad() #勾配を0で初期化
        loss = creterion(model(X), y)
        loss.backward()
        optimizer.step() #パラメータ更新

    with torch.no_grad(): #テンソルの勾配の計算を不可
        y_pred = model(X_train)
        loss = creterion(y_pred, y_train)
        acc = accuracy(y_pred, y_train)
        train_loss.append(loss)
        train_acc.append(acc)

        y_pred = model(X_valid)
        loss = creterion(y_pred, y_valid)
        acc = accuracy(y_pred, y_valid)
        valid_loss.append(loss)
        valid_acc.append(acc)

    #モデルの保存
    torch.save(model.state_dict(), f'./model76/{epoch}')