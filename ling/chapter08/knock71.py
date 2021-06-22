import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__=="__main__":
    #load the saved tensor, size:10672 x 300
    X_train=torch.load("./X_train.pt")
    
    '''
    input vector size: 1x300
    output size: 1x4
    '''
    model=Net(300,4)

    y_h_1=torch.softmax(model(X_train[:1]),dim=-1)

    print(y_h_1)

    Y_h=torch.softmax(model.forward(X_train[:4]),dim=-1)

    print(Y_h)

    torch.save(model,'./model.pt')