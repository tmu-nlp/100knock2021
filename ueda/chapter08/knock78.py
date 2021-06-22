batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for b in batch_size:
    device = torch.device('cuda')
    model = LogisticNet()
    dataset = create_data(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size = b, shuffle = True)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    train_accuracy = []
    valid_accuracy = []
    train_loss = []
    valid_loss = []
    for epoch in tqdm(range(5)):
        for inputs, target in train_loader:
            optim.zero_grad()
            model.to(device)
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, target)
            loss.backward()
            optim.step()
            torch.save({'epoch':epoch, 'model':model.state_dict(), 'optim':optim.state_dict()}, f'C:\Git\checkpoints\checkpoint_{epoch+1}.pt')
        with torch.no_grad():
            x_train = x_train.to(device)
            x_valid = x_valid.to(device)
            pred = model(x_train)
            train_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_train))
            train_loss.append(loss_func(pred, y_train))
            pred = model(x_valid)
            valid_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_valid))
            valid_loss.append(loss_func(pred, y_valid))
#batch_size=1 00:56
#batch_size=2 00:28
#batch_size=4 00:14
#batch_size=8 00:07
#batch_size=16 00:03
#batch_size=32 00:01
#batch_size=64 00:01
#batch_size=128 00:00
#batch_size=256 00:00
#batch_size=512 00:00
#batch_size=1024 00:00