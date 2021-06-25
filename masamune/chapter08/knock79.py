import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time

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
        self.layer1 = torch.nn.Linear(input_size, 900)
        self.layer2 = torch.nn.Linear(900, 450)
        self.layer3 = torch.nn.Linear(450, output_size)
        torch.nn.init.normal_(self.layer1.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, input):
        softmax = torch.nn.Softmax(dim=-1)
        ReLU = torch.nn.ReLU()
        Dropout = torch.nn.Dropout(p=0.5)

        x = ReLU(self.layer1(input))
        x = Dropout(x)
        x = ReLU(self.layer2(x))
        x = Dropout(x)
        x = self.layer3(x)
        output = softmax(x)

        return output

#GPUの指定
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
model = NN(300, 4).to(device)
#(X, y)の組を作成
data_train = TensorDataset(X_train, y_train)
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
dataloader = DataLoader(data_train, batch_size=100, shuffle=True)
for epoch in range(150):
    #計測開始
    start = time.time()
    for X, y in dataloader:
        optimizer.zero_grad() #勾配を0で初期化
        y_pred = model(X.to(device))
        loss = creterion(y_pred, y.to(device)) #モデルと同じデバイス
        loss.backward()
        optimizer.step() #パラメータ更新
    #計測終了
    t = time.time() - start

    with torch.no_grad(): #テンソルの勾配の計算を不可
        y_pred = model(X_train.to(device))
        train_loss = creterion(y_pred, y_train)
        train_acc = accuracy(y_pred, y_train)

        y_pred = model(X_valid.to(device))
        valid_loss = creterion(y_pred, y_valid)
        valid_acc = accuracy(y_pred, y_valid)

torch.save(model.state_dict(), './model79.pth')
print(f'train_loss：{train_loss}, train_accuracy：{train_acc}, valid_loss：{valid_loss}, valid_accuracy：{valid_acc}')
print(f'time：{t} sec')

'''
train_loss：0.8344430327415466, train_accuracy：0.9095764617691154, valid_loss：0.8748437762260437, valid_accuracy：0.8673163418290855
time：0.7924339771270752 sec
'''