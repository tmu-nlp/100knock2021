import torch
import numpy as np

X_train = np.loadtxt('./data/X_train.txt', delimiter=' ')
X_train = torch.tensor(X_train, dtype=torch.float32)
W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim=1)
print (softmax(torch.matmul(X_train[:1], W)))
print (softmax(torch.matmul(X_train[:4], W)))