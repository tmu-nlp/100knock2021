from matplotlib import pyplot as plt

model = LogisticNet()
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, shuffle = True)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []
for epoch in tqdm(range(20)):
    for inputs, target in train_loader:
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optim.step()
    with torch.no_grad():
        pred = model(x_train)
        train_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_train))
        train_loss.append(loss_func(pred, y_train))
        pred = model(x_valid)
        valid_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_valid))
        valid_loss.append(loss_func(pred, y_valid))

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.plot(train_accuracy, label='train')
plt.plot(valid_accuracy, label='valid')
plt.subplot(1,2,2)
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='valid')
plt.show()