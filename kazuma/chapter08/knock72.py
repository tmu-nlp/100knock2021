import torch
import numpy as np
def knock72():
    X_train = np.loadtxt("data/train_fea.txt", delimiter=' ')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = np.loadtxt('data/train_label.txt')
    y_train = torch.tensor(y_train, dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()
    W = torch.randn(300, 4)
    softmax = torch.nn.Softmax(dim=1)
    print (loss(torch.matmul(X_train[:1], W),y_train[:1]))
    print (loss(torch.matmul(X_train[:4], W),y_train[:4]))

    ans = [] 
    for s,i in zip(softmax(torch.matmul(X_train[:4], W)),y_train[:4]):
        ans.append(-np.log(s[i]))
    print (np.mean(ans))

if __name__ == "__main__":
    knock72()