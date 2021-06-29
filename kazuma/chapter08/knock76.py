from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from knock74 import accuracy
from knock73 import LogisticRegression

def calculate_loss_sum(loader, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_sum = 0
    cnt = 0
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        loss_sum += loss
        cnt += 1
    return loss_sum/cnt

def calculate_accuracy(model, y, X):
    pred_train = model(X)
    return accuracy(pred_train, y)


def knock76():
    X_train = torch.tensor(np.loadtxt("data/train_fea.txt", delimiter=' '), dtype=torch.float32)
    y_train = torch.tensor(np.loadtxt('data/train_label.txt'), dtype=torch.int64)
    X_valid = torch.tensor(np.loadtxt("data/valid_fea.txt", delimiter=' '), dtype=torch.float32)
    y_valid = torch.tensor(np.loadtxt('data/valid_label.txt'), dtype=torch.int64)

    loss = torch.nn.CrossEntropyLoss()
    model = LogisticRegression()
    ds_train = TensorDataset(X_train, y_train)
    # DataLoaderを作成
    loader_train = DataLoader(ds_train, batch_size=1, shuffle=True)
    ds_valid = TensorDataset(X_valid, y_valid)
    # DataLoaderを作成
    loader_valid = DataLoader(ds_valid, batch_size=1, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    y1_t, y1_v, y2_t, y2_v, x = [],[],[],[],[]

    for epoch in range(10):
        train_loss_sum = 0
        cnt = 0
        for xx, yy in loader_train:
            y_pred = model(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss
            cnt += 1
        
        acc_train = calculate_accuracy(model, y_train, X_train)
        valid_loss_sum = calculate_loss_sum(loader_valid, model)
        acc_valid = calculate_accuracy(model, y_valid, X_valid)
        x += [epoch]
        y1_t += [train_loss_sum.item()/cnt]
        y1_v += [valid_loss_sum.item()]
        y2_t += [acc_train]
        y2_v += [acc_valid]
        torch.save(model.state_dict(), f"knock76_output/{epoch}.model")
        torch.save(optimizer.state_dict(), f"knock76_output/{epoch}.param")
        
    ax1.plot(x, y1_t, color = "blue", label = "loss_train")
    ax1.plot(x, y1_v, color = "orange", label = "loss_valid")
    ax2.plot(x, y2_t, color = "blue", label = "acc_train")
    ax2.plot(x, y2_v, color = "orange", label = "acc_valid")
    ax1.legend(loc = 'upper right') #凡例
    ax2.legend(loc = 'upper right') #凡例
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    knock76() 