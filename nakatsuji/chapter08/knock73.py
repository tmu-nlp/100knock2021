from torch.utils.data import TensorDataset, DataLoader
from torch import nn

class SNN(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.fc = nn.Linear(in_d, out_d, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        return self.fc(x)

model = SNN(300, 4)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, shuffle=True)

los_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
    for X, y in dataloader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fun(outputs, y)
        loss.backward()
        optimizer.step()
    
