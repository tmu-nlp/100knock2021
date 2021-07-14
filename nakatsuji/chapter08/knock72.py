y_train = torch.load(path + 'y_train.pt')
loss_fun = torch.nn.CrossEntropyLoss()
print(loss_fun(torch.matmul(X_train[:1], W), y_train[:1]))
print(loss_fun(torch.matmul(X_train[:4], W), y_train[:4]))