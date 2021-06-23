import torch

X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')
W = torch.randn(300, 4)

loss = torch.nn.CrossEntropyLoss()
l1 = loss(torch.matmul(X_train[:1], W), y_train[:1])
l = loss(torch.matmul(X_train[:4], W), y_train[:4])
print(l1)
print(l)

'''

tensor(1.5166)
tensor(2.2556)

'''