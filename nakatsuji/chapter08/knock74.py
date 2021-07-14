def acc(pred, y):
    cnt = 0
    for a_pred, a_y in zip(pred, y):
        if a_pred == a_y:
            cnt += 1
    return cnt/len(pred)


X_valid = torch.load(path + 'X_valid.pt')
y_valid = torch.load(path + 'y_valid.pt')
train_pred = model(X_train)
valid_pred = model(X_valid)
print(acc(torch.argmax(train_pred, dim=1), y_train))
print(acc(torch.argmax(valid_pred, dim=1), y_valid))
