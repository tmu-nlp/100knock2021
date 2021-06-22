y_train = torch.load(r'C:\Git\y_train.pt')
c_loss = torch.nn.CrossEntropyLoss()
print(c_loss(torch.matmul(x_train[:1], W), y_train[:1]))
print(c_loss(torch.matmul(x_train[:4], W), y_train[:4]))