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
        self.layer1 = torch.nn.Linear(input_size, output_size, bias=False) #全結合層の定義
        torch.nn.init.normal_(self.layer1.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, input):
        activation = torch.nn.Softmax(dim=-1)
        output = activation(self.layer1(input))

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
batch_size = [2**x for x in range(10)]
for batch in batch_size:
    dataloader = DataLoader(data_train, batch_size=batch, shuffle=True)
    for epoch in range(1):
        #計測開始
        start = time.time()
        for X, y in dataloader:
            optimizer.zero_grad() #勾配を0で初期化
            loss = creterion(model(X.to(device)), y.to(device)) #モデルと同じデバイス
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

    print(f'batch_size：{batch} time：{t} sec')
    print(f'train_loss：{train_loss}, train_accuracy：{train_acc}, valid_loss：{valid_loss}, valid_accuracy：{valid_acc}')
    print()

'''
batch_size：1 time：2.6055009365081787 sec
train_loss：0.9697434902191162, train_accuracy：0.7778298350824587, valid_loss：0.9724473357200623, valid_accuracy：0.7758620689655172

batch_size：2 time：1.458482265472412 sec
train_loss：0.9598569869995117, train_accuracy：0.7861694152923538, valid_loss：0.9630810618400574, valid_accuracy：0.7811094452773614

batch_size：4 time：0.8322741985321045 sec
train_loss：0.9551302790641785, train_accuracy：0.7910419790104948, valid_loss：0.9585632681846619, valid_accuracy：0.787856071964018

batch_size：8 time：0.381929874420166 sec
train_loss：0.952693521976471, train_accuracy：0.7931034482758621, valid_loss：0.9563083648681641, valid_accuracy：0.7938530734632684

batch_size：16 time：0.2256026268005371 sec
train_loss：0.9514397978782654, train_accuracy：0.7945089955022488, valid_loss：0.9551610946655273, valid_accuracy：0.7938530734632684

batch_size：32 time：0.14585185050964355 sec
train_loss：0.9508019089698792, train_accuracy：0.7950712143928036, valid_loss：0.9545855522155762, valid_accuracy：0.7946026986506747

batch_size：64 time：0.10264086723327637 sec
train_loss：0.9504820704460144, train_accuracy：0.7952586206896551, valid_loss：0.954300045967102, valid_accuracy：0.7946026986506747

batch_size：128 time：0.07994794845581055 sec
train_loss：0.9503207802772522, train_accuracy：0.7954460269865068, valid_loss：0.9541563987731934, valid_accuracy：0.795352323838081

batch_size：256 time：0.07310104370117188 sec
train_loss：0.950239896774292, train_accuracy：0.7955397301349325, valid_loss：0.9540846347808838, valid_accuracy：0.795352323838081

batch_size：512 time：0.08801984786987305 sec
train_loss：0.950199544429779, train_accuracy：0.7955397301349325, valid_loss：0.9540489315986633, valid_accuracy：0.7946026986506747
'''