import numpy as np
import torch

X_train = np.loadtxt('./data/X_train.txt', delimiter=',')
X_train = torch.tensor(X_train, dtype=torch.float32) 

W = torch.randn(300, 4) #標準正規分布からの乱数 shape = (300, 4)

softmax = torch.nn.Softmax(dim=1)
y = softmax(torch.matmul(X_train[:1], W))
Y = softmax(torch.matmul(X_train[:4], W))
print(y)
print(Y)

'''
出力
tensor([[0.0687, 0.7204, 0.0591, 0.1518]])
tensor([[0.0687, 0.7204, 0.0591, 0.1518],
        [0.2142, 0.4404, 0.1601, 0.1854],
        [0.0154, 0.5563, 0.0184, 0.4099],
        [0.4216, 0.4460, 0.1118, 0.0206]])
'''