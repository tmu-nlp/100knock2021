import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import tqdm
import time


'''
no cuda
no gpu support for m1 mac
executed in google colab
'''
class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        
        #define layer
        self.l1=nn.Linear(input_size,output_size,bias=False)
        #initalize weight of l1 in normal distribution
        nn.init.normal_(self.l1.weight,0.0,1.0)

    def forward(self,x):
        x=self.l1(x)
        return x


class Newdataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]


def loss_accuracy(model,criterion,loader,device):
    model.eval()
    loss=0.0
    total=0
    correct=0

    with torch.no_grad():
        for inputs,labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs=model(inputs)
            loss+=criterion(outputs,labels).item()
            pred=torch.argmax(outputs,dim=-1)
            total+=len(inputs)
            correct+=(pred==labels).sum().item()
    return loss/len(loader),correct/total

def train_model(DS_train,DS_valid,batch_size,model,criterion,optimizer,num_epochs,device=None):
    model.to(device)
    
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        
        loss_train,acc_train=loss_accuracy(model,criterion,DL_train,device)
        loss_valid,acc_valid=loss_accuracy(model,criterion,DL_valid,device)

        loss_train_valid.append([loss_train,loss_valid])
        acc_train_valid.append([acc_train,acc_valid])

        #torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        end=time.time()
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f},loss_valid: {loss_valid:.4f},accuracy_valid: {acc_valid:.4f},{(end - start):.4f}sec')
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

device=torch.device('')

num_epochs=5

for batch_size in [2**i for i in range(11)]:
    print('batch size: '+str(batch_size))
    log=train_model(DS_train,DS_valid,batch_size,model,criterion,optimizer,num_epochs,device)
