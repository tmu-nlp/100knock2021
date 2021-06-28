from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import time
path = "/content/drive/MyDrive/basis2021/nlp100/"
X_train = torch.load(path + 'X_train.pt')
y_train = torch.load(path + 'y_train.pt')
X_valid = torch.load(path + 'X_valid.pt')
y_valid = torch.load(path + 'y_valid.pt')




#モデル
torch.manual_seed(0)
model = SNN(300, 4)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

batch_sizes = [1, 2, 4, 8, 16, 32]
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []

for a_batch in batch_sizes:

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size= a_batch, shuffle=True)
    start = time.time()
    for epoch in tqdm(range(1)):
        for X, y in dataloader:
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fun(out, y)
            loss.backward()
            optimizer.step()
    end = time.time()
    print(f'batch_size:{a_batch} >>> {end - start}')
    with torch.no_grad():
        pred = model(X_train)
        train_loss.append(loss_fun(pred, y_train))
        train_acc.append(acc(torch.argmax(pred, dim=1), y_train))
        pred = model(X_valid)
        valid_loss.append(loss_fun(pred, y_valid))
        valid_acc.append(acc(torch.argmax(pred, dim=1), y_valid))

        #print(f'epoch{epoch+1}, train_loss:{train_loss[epoch]}, train_acc:{train_acc[epoch]}, valid_loss:{valid_loss[epoch]}, valid_acc:{valid_acc[epoch]}')


#plt.plot(train_loss, label="train loss")
#plt.plot(valid_loss, label="valid loss")
#plt.legend()
#plt.show()
#
#plt.plot(train_acc, label="train acc")
#plt.plot(valid_acc, label="valid acc")
#plt.legend()
#plt.show()

'''
100%|██████████| 1/1 [00:02<00:00,  2.34s/it]
  0%|          | 0/1 [00:00<?, ?it/s]batch_size:1 >>> 2.3461833000183105
100%|██████████| 1/1 [00:01<00:00,  1.30s/it]
  0%|          | 0/1 [00:00<?, ?it/s]batch_size:2 >>> 1.3072640895843506
100%|██████████| 1/1 [00:00<00:00,  1.43it/s]
  0%|          | 0/1 [00:00<?, ?it/s]batch_size:4 >>> 0.7016232013702393
100%|██████████| 1/1 [00:00<00:00,  2.60it/s]
  0%|          | 0/1 [00:00<?, ?it/s]batch_size:8 >>> 0.3871943950653076
100%|██████████| 1/1 [00:00<00:00,  4.25it/s]
  0%|          | 0/1 [00:00<?, ?it/s]batch_size:16 >>> 0.23801326751708984
100%|██████████| 1/1 [00:00<00:00,  6.51it/s]batch_size:32 >>> 0.1569817066192627
'''