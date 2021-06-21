x_train = torch.load(r'C:\Git\x_train.pt')
W = 0.2*torch.randn(300, 4)-0.1
softmax = torch.nn.Softmax(dim=1)
print(softmax(torch.matmul(x_train[:1], W)))
print(softmax(torch.matmul(x_train[:4], W)))