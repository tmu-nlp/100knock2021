import numpy as np
from knock71 import Net
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import time

class MultiLayerNet(nn.Module):
    def __init__(self,input_size,mid_size,output_size,mid_layers):
        super().__init__()
        self.mid_layers=mid_layers
        #define layer

        self.l1=nn.Linear(input_size,mid_size)
        self.l2=nn.Linear(mid_size,mid_size)
        self.l3=nn.Linear(mid_size,output_size)
        #initalize weight of l1 in normal distribution
        self.bn=nn.BatchNorm1d(mid_size)

    def forward(self,x):
        x=F.relu(self.l1(x))
        for _ in range(self.mid_layers):
            x=F.relu(self.bn(self.l2(x)))
        x=F.relu(self.l3(x))
        return x


class Newdataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]


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

def train_model(DS_train,DS_valid,batch_size,model,criterion,optimizer,num_epochs):
    DL_train=DataLoader(DS_train,batch_size=batch_size,shuffle=True)
    DL_valid=DataLoader(DS_valid,batch_size=len(DS_valid),shuffle=False)

    loss_train_valid=[]
    acc_train_valid=[]

    
    for epoch in range(num_epochs):
    #set the module in training mode 
        start=time.time()
        model.train()
        for i ,(inputs,labels) in enumerate(DL_train):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        
        loss_train,acc_train=loss_accuracy(model,criterion,DL_train)
        loss_valid,acc_valid=loss_accuracy(model,criterion,DL_valid)

        loss_train_valid.append([loss_train,loss_valid])
        acc_train_valid.append([acc_train,acc_valid])

        #torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        end=time.time()
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f},acc_train: {acc_train:.4f},accuracy_valid: {acc_valid:.4f},{(end - start):.4f}sec')
    return {'loss of train and valid:':loss_train_valid,'accuracy of train & valid':acc_train_valid}

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

model=Net(300,4)

criterion=nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr=1e-1)

num_epochs=100

batch_size=64

log=train_model(DS_train,DS_valid,batch_size,model,criterion,optimizer,num_epochs)
