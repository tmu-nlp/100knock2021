import torch
import numpy as np
from pprint import pprint

def knock71():
    X_train = np.loadtxt("data/train_fea.txt", delimiter=' ')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    print(X_train)
    W = torch.randn(300, 4)
    softmax = torch.nn.Softmax(dim=1)
    print (softmax(torch.matmul(X_train[:1], W)))
    print (softmax(torch.matmul(X_train[:4], W)))
    

if __name__ == "__main__":
    knock71()