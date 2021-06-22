import numpy as np
from knock71 import Net
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

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

'''
In each epoch, the model will train firstly, then calculate the loss, 
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

        loss_train+=loss.item()
    
    #calculate average loss for each batch
    loss_train=loss_train/i

    #sets the module in evaluation mode
    model.eval()
    with torch.no_grad():
        inputs,labels=next(iter(DL_valid))
        outputs=model(inputs)
        loss_valid=criterion(outputs,labels)
    
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')

torch.save(model,"./model_trained_73.pt")