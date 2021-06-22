import torch
import torch.nn as nn
from knock71 import Net 

criterion=nn.CrossEntropyLoss()

model=torch.load("./model.pt")
X_train=torch.load("./X_train.pt")
y_train=torch.load("./y_train.pt")
#
l_1=criterion(model(X_train[:1]),y_train[:1])

model.zero_grad()
l_1.backward()

print(l_1)
print(model.l1.weight.grad)

l=criterion(model(X_train[:4]),y_train[:4])

model.zero_grad()
l.backward()

print(l)
print(model.l1.weight.grad)