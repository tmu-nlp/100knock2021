from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm_notebook as tqdm


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)


X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')

model = LogisticRegression() #モデルの定義
ds = TensorDataset(X_train, y_train) #Datasetの定義
loader = DataLoader(ds, batch_size=1, shuffle=True) # DataLoaderを作成
loss_fn = torch.nn.CrossEntropyLoss() #損失関数の定義
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1) #オプティマイザの定義

for epoch in range(tqdm(10)):
    for xx, yy in loader:
        optimizer.zero_grad() #勾配を0で初期化
        y_pred = model(xx) #categoryの予測
        loss = loss_fn(y_pred, yy) #損失率計算
        loss.backward() #誤差逆伝播
        optimizer.step() #重み更新