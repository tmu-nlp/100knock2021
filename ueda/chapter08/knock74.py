def calc_accuracy(pred, answer):
    correct = 0
    for i in range(len(answer)):
        if pred[i] == answer[i]: correct+=1
    return correct/len(answer)

x_valid = torch.load(r'C:\Git\x_valid.pt')
y_valid = torch.load(r'C:\Git\y_valid.pt')
pred = model(x_train)
print(calc_accuracy(torch.argmax(pred, dim=1), y_train))
pred = model(x_valid)
print(calc_accuracy(torch.argmax(pred, dim=1), y_valid))

#0.9119243728940472
#0.8913857677902621