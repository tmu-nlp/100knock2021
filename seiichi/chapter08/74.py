from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

X_train = np.loadtxt('./data/X_train.txt', delimiter=' ')
X_train = torch.tensor(X_train, dtype=torch.float32)
W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim=1)

y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model_path = './bin/model.bin'
model = LogisticRegression()
model.load_state_dict(torch.load(model_path))

def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)
    label = label.data.numpy()
    return (pred == label).mean()

X_valid = np.loadtxt('./data/X_valid.txt', delimiter=' ')
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = np.loadtxt('./data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)

pred = model(X_train)
print (accuracy(pred, y_train))
pred = model(X_valid)
print (accuracy(pred, y_valid))