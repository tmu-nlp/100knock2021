import numpy as np
from knock71 import Net
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

'''
@para
model
criterion
loader

@return
loss
accuracy
'''
def loss_accuracy(model,criterion,loader):
    model.eval()
    loss=0.0
    total=0
    correct=0

    with torch.no_grad():
        for inputs,labels in loader:
            outputs=model(inputs)
            loss+=criterion(outputs,labels).item()
            pred=torch.argmax(outputs,dim=-1)
            total+=len(inputs)
            correct+=(pred==labels).sum().item()
    return loss/len(loader),correct/total

class Newdataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]

#read data 
X_train=torch.load("./X_train.pt")
y_train=torch.load("./y_train.pt")
X_test=torch.load("./X_test.pt")
y_test=torch.load("./y_test.pt")
X_valid=torch.load("./X_valid.pt")
y_valid=torch.load("./y_valid.pt")

#DS for Dataset
DS_train = Newdataset(X_train,y_train)
DS_valid = Newdataset(X_valid,y_valid)
DS_test = Newdataset(X_test,y_test)

#create dataloader,DL for DataLoader
DL_train = DataLoader(DS_train,batch_size=1,shuffle=True)
DL_valid = DataLoader(DS_valid,batch_size=len(DS_valid),shuffle=False)
DL_test = DataLoader(DS_test,batch_size=len(DS_test),shuffle=False)

model=Net(300,4)
#set CrossEntropyLoss to calculate loss
criterion=nn.CrossEntropyLoss()
#set SGD as optimizer,lr stands for learning rate
optimizer = torch.optim.SGD(model.parameters(),lr=1e-1)
#define number of epochs
num_epochs=10
loss_train_valid=[]
acc_train_valid=[]
'''

'''
for epoch in range(num_epochs):
    #set the module in training mode 
    model.train()
    loss_train=0.0
    for i ,(inputs,labels) in enumerate(DL_train):
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
    
    loss_train,acc_train=loss_accuracy(model,criterion,DL_train)
    loss_valid,acc_vaild=loss_accuracy(model,criterion,DL_valid)

    loss_train_valid.append([loss_train,loss_valid])
    acc_train_valid.append([acc_train,acc_vaild])

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')
