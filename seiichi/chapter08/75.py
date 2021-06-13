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

train_loss, val_loss = [], []
train_acc, val_acc = [], []

track = False
save = True

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss.append(loss)
        train_acc.append(accuracy(y_pred,y_train))
        y_pred = model(X_valid)
        loss = loss_fn(y_pred, y_valid)
        val_loss.append(loss)
        val_acc.append(accuracy(y_pred,y_valid))
    if track:
        fig = plt.figure()
        plt.plot(range(epoch+1), train_loss, val_loss)
        plt.savefig("./fig/loss_{}.png".format(epoch))
        fig = plt.figure()
        plt.plot(range(epoch+1), train_acc, val_acc)
        plt.savefig("./fig/acc_{}.png".format(epoch))