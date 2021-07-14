from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
path = "/content/drive/MyDrive/basis2021/nlp100/"
X_train = torch.load(path + 'X_train.pt')
y_train = torch.load(path + 'y_train.pt')
X_valid = torch.load(path + 'X_valid.pt')
y_valid = torch.load(path + 'y_valid.pt')

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, shuffle=True)

train_loss, train_acc = [], []
valid_loss, valid_acc = [], []

#モデル
torch.manual_seed(0)
model = SNN(300, 4)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)


for epoch in tqdm(range(10)):
    for X, y in dataloader:
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fun(out, y)
        loss.backward()
        optimizer.step()

    joblib.dump(model.state_dict(), f"checkpoint_{epoch}.joblib")

    with torch.no_grad():
        pred = model(X_train)
        train_loss.append(loss_fun(pred, y_train))
        train_acc.append(acc(torch.argmax(pred, dim=1), y_train))
            
        pred = model(X_valid)
        valid_loss.append(loss_fun(pred, y_valid))
        valid_acc.append(acc(torch.argmax(pred, dim=1), y_valid))
    
    print(f'epoch{epoch+1}, train_loss:{train_loss[epoch]}, train_acc:{train_acc[epoch]}, valid_loss:{valid_loss[epoch]}, valid_acc:{valid_acc[epoch]}')


plt.plot(train_loss, label="train loss")
plt.plot(valid_loss, label="valid loss")
plt.legend()
plt.show()

plt.plot(train_acc, label="train acc")
plt.plot(valid_acc, label="valid acc")
plt.legend()
plt.show()