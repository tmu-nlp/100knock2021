import numpy as np
import torch

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

print('[x1]')
model = NN(300, 4)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(model(X_train[:1]), y_train[:1])
model.zero_grad() #勾配を0で初期化
loss.backward() #勾配を計算
print(f'cross entropy：{loss}')
print(f'grads：{model.layer1.weight.grad}')

print('[x1, x2, x3, x4]')
model = NN(300, 4)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad() #勾配を0で初期化
loss.backward() #勾配を計算
print(f'cross entropy：{loss}')
print(f'grads：{model.layer1.weight.grad}')

'''
[x1]
cross entropy：1.335367202758789
grads：tensor([[ 4.3471e-02,  2.0887e-03,  1.8855e-02,  ..., -4.4732e-03,
          3.4443e-04,  2.2635e-02],
        [-2.9519e-02, -1.4183e-03, -1.2804e-02,  ...,  3.0375e-03,
         -2.3388e-04, -1.5370e-02],
        [-1.2461e-02, -5.9870e-04, -5.4048e-03,  ...,  1.2822e-03,
         -9.8729e-05, -6.4883e-03],
        [-1.4914e-03, -7.1657e-05, -6.4689e-04,  ...,  1.5346e-04,
         -1.1817e-05, -7.7657e-04]])
[x1, x2, x3, x4]
cross entropy：1.5419368743896484
grads：tensor([[ 9.2179e-03, -4.3319e-03, -3.8405e-03,  ..., -3.1553e-04,
         -4.4890e-03,  3.4833e-03],
        [-2.1869e-04,  2.1738e-04,  9.8699e-04,  ..., -9.7083e-05,
          1.0730e-03, -2.7749e-04],
        [-6.1790e-03,  9.7825e-04,  4.3229e-05,  ...,  1.5334e-03,
          1.0298e-03, -1.9189e-03],
        [-2.8202e-03,  3.1363e-03,  2.8103e-03,  ..., -1.1208e-03,
          2.3862e-03, -1.2869e-03]])
'''