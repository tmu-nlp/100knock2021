from knock73 import DL_test, DL_train
import torch


def accuracy(model,loader):
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for inputs,labels in loader:
            outputs=model(inputs)
            pred=torch.argmax(outputs,dim=-1)
            total+=len(inputs)
            correct+=(pred==labels).sum().item()
    return correct/total
    
model=torch.load("./model_trained_73.pt")

acc_train=accuracy(model,DL_train)
acc_test=accuracy(model,DL_test)
print(acc_train)
print(acc_test)