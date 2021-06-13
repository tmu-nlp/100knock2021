import time
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('./data/X_train.txt', delimiter=' ')
X_train = torch.tensor(X_train, dtype=torch.float32)
W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim=1)

y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)

X_valid = np.loadtxt('./data/X_valid.txt', delimiter=' ')
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = np.loadtxt('./data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)
    label = label.data.numpy()
    return (pred == label).mean()

model = LogisticRegression()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

ls_bs = [2**i for i in range(15)]
ls_time = []
for bs in ls_bs:
  loader = DataLoader(ds, batch_size=bs, shuffle=True)
  optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)
  for epoch in range(1):
      start = time.time()
      for xx, yy in loader:
          y_pred = model(xx)
          loss = loss_fn(y_pred, yy)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  ls_time.append(time.time()-start)
print(ls_time)