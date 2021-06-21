from torch.utils.data import DataLoader
from tqdm import tqdm

class LogisticNet(torch.nn.Module):
    def __init__(self, D_in=300, D_out=4):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)
        torch.nn.init.normal_(self.linear.weight, -0.1, 0.1)
    def forward(self, x):
        return self.linear(x)

def create_data(x, y):
    data = []
    for i in range(len(y)):
        data.append([x[i], y[i]])
    return data
    
model = LogisticNet()
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, shuffle = True)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in tqdm(range(100)):
    for inputs, target in train_loader:
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optim.step()