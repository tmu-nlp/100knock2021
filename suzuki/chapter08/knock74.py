#knock74
import numpy as np
import torch

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1) #各単語ベクトルの要素の最大値
  label = label.data.numpy() #ラベルをpredと比べられる形にしてる?
  return (pred == label).mean() #正解率

X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')
model = torch.load('model.pt')

pred = model(X_train)
print(accuracy(pred, y_train))

pred = model(X_valid)
print(accuracy(pred, y_valid))

'''

0.9220329464619993
0.8989520958083832

'''