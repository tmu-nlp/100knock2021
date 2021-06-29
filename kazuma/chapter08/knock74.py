import numpy as np
import torch
import pickle
from knock73 import LogisticRegression

def accuracy(pred, ans):
    pred = np.argmax(pred.data.numpy(), axis = 1)
    ans = ans.data.numpy()
    acc = (pred == ans).mean()
    return acc

def get_trained_model():
    with open("trained_model.pickle", "rb") as f1:
        trained_model = pickle.load(f1)
        return trained_model

def knock74():
    model = get_trained_model()
    X_train = np.loadtxt("data/train_fea.txt", delimiter=' ')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = np.loadtxt('data/train_label.txt')
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_valid = np.loadtxt("data/valid_fea.txt", delimiter=' ')
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = np.loadtxt('data/valid_label.txt')
    y_valid = torch.tensor(y_valid, dtype=torch.int64)

    pred_train = model(X_train)
    print(accuracy(pred_train, y_train))
    pred_valid = model(X_valid)
    print(accuracy(pred_valid, y_valid))

if __name__ == "__main__":
    knock74()