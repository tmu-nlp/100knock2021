import torch.nn.functional as F
class LogisticNet(torch.nn.Module):
    def __init__(self, D_in=300, D_mid=256, D_out=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_in, D_mid)
        self.fc2 = torch.nn.Linear(D_mid, D_mid)
        self.fc3 = torch.nn.Linear(D_mid, D_out)
        self.dropout1 = torch.nn.Dropout2d(0.1)
        self.dropout2 = torch.nn.Dropout2d(0.1)
        torch.nn.init.normal_(self.fc1.weight, -0.1, 0.1)
        torch.nn.init.normal_(self.fc2.weight, -0.1, 0.1)
        torch.nn.init.normal_(self.fc3.weight, -0.1, 0.1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return  F.relu(self.fc3(x))

device = torch.device('cuda')
model = LogisticNet()
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()
train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []
for epoch in tqdm(range(50)):
    for inputs, target in train_loader:
        optim.zero_grad()
        model.to(device)
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optim.step()
        #torch.save({'epoch':epoch, 'model':model.state_dict(), 'optim':optim.state_dict()}, f'C:\Git\checkpoints\checkpoint_{epoch+1}.pt')
    with torch.no_grad():
        x_train = x_train.to(device)
        x_valid = x_valid.to(device)
        y_train = y_train.to(device)
        y_valid = y_valid.to(device)
        pred = model(x_train)
        print(pred)
        train_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_train))
        train_loss.append(loss_func(pred, y_train).detach().cpu().numpy())
        pred = model(x_valid)
        valid_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_valid))
        valid_loss.append(loss_func(pred, y_valid).detach().cpu().numpy())
plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.plot(train_accuracy, label='train')
plt.plot(valid_accuracy, label='valid')
plt.subplot(1,2,2)
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='valid')
plt.show()