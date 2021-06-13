import torch
import numpy as np

X_train = np.loadtxt('./data/X_train.txt', delimiter=' ')
X_train = torch.tensor(X_train, dtype=torch.float32)
W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim=1)

y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
loss = torch.nn.CrossEntropyLoss()
print (loss(torch.matmul(X_train[:1], W),y_train[:1]))
print (loss(torch.matmul(X_train[:4], W),y_train[:4]))