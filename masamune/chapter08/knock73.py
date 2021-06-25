import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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
#DataLoaderの作成
dataloader = DataLoader(data_train, batch_size=1, shuffle=True)
creterion = torch.nn.CrossEntropyLoss()
#最適化アルゴリズムの定義
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#学習
for epoch in range(10):
    for X, y in dataloader:
        optimizer.zero_grad() #勾配を0で初期化
        loss = creterion(model(X), y)
        loss.backward()
        optimizer.step() #パラメータ更新

print(model.layer1.weight.grad)
torch.save(model.state_dict(), './model73.pth') #モデル保存

'''
tensor([[-4.2115e-07,  1.7470e-06, -1.6656e-06,  ..., -6.0850e-08,
          1.5784e-06, -1.5585e-06],
        [-3.5763e-03,  1.4836e-02, -1.4144e-02,  ..., -5.1673e-04,
          1.3404e-02, -1.3235e-02],
        [ 3.5768e-03, -1.4837e-02,  1.4146e-02,  ...,  5.1679e-04,
         -1.3405e-02,  1.3236e-02],
        [-1.8824e-08,  7.8087e-08, -7.4448e-08,  ..., -2.7198e-09,
          7.0550e-08, -6.9660e-08]])
'''