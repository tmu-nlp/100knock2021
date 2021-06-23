import torch

X_train = torch.load('X_train.pt')
W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim = 1)
y1 = softmax(torch.matmul(X_train[:1], W))
Y = softmax(torch.matmul(X_train[:4], W))
print(y1)
print(Y)

'''

tensor([[0.0016, 0.2433, 0.2929, 0.4622]])
tensor([[0.0016, 0.2433, 0.2929, 0.4622],
        [0.0286, 0.1463, 0.1549, 0.6703],
        [0.4086, 0.0342, 0.3137, 0.2435],
        [0.0227, 0.0664, 0.0632, 0.8477]])

'''