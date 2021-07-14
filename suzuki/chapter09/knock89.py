import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import *
import torch.nn as nn
import torch.nn.functional as F

max_len = 15
PAD = 0
n_unit =  768

tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class BertClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super(BertClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased') 
        self.fc = nn.Linear(n_unit, n_classes)

    def forward(self, ids):
        seg_ids = torch.zeros_like(ids) # 全て同一セグメントとみなす
        attention_mask = (ids > 0)
        last_hidden_state, _ = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        x = last_hidden_state[:,0,:] # CLSトークン
        logit = self.fc(x.view(-1,n_unit))
        return logit


def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()

def df2id(df):
  tokenized = df[1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  return tokenized

train = pd.read_csv('train.txt', header=None, sep='\t') 
valid = pd.read_csv('valid.txt', header=None, sep='\t') 
test = pd.read_csv('test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')
y_test = torch.load('y_test.pt')

model = BertClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dfs_freeze(model)
model.fc.requires_grad_(True)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(30):
    print(epoch)
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device)) 
        print ('epoch: {}, Loss/train: {}, Acc/train: {}'.format(epoch, loss.item(), accuracy(y_pred,y_train)))

        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print ('epoch: {}, Loss/train: {}, Acc/train: {}'.format(epoch, loss.item(), accuracy(y_pred,y_valid)))

'''

epoch: 0, Loss/train: 0.9545313715934753, Acc/train: 0.6859790340696369

epoch: 1, Loss/train: 0.7911579608917236, Acc/train: 0.7544926993635342

epoch: 2, Loss/train: 0.6911827921867371, Acc/train: 0.7741482590789966

epoch: 3, Loss/train: 0.6274504661560059, Acc/train: 0.7882815424934482

epoch: 4, Loss/train: 0.5825473070144653, Acc/train: 0.8037251965555972

epoch: 5, Loss/train: 0.5496016144752502, Acc/train: 0.8141145638337701

epoch: 6, Loss/train: 0.5241172313690186, Acc/train: 0.8242231374017222

epoch: 7, Loss/train: 0.5040343403816223, Acc/train: 0.8290902283788844

epoch: 8, Loss/train: 0.48729759454727173, Acc/train: 0.8339573193560464

epoch: 9, Loss/train: 0.47349077463150024, Acc/train: 0.8389180082366156

epoch: 10, Loss/train: 0.46156758069992065, Acc/train: 0.8412579558217896

epoch: 11, Loss/train: 0.45109784603118896, Acc/train: 0.8450954698614751

epoch: 12, Loss/train: 0.44211795926094055, Acc/train: 0.8474354174466492

epoch: 13, Loss/train: 0.434067040681839, Acc/train: 0.8501497566454511

epoch: 14, Loss/train: 0.426910400390625, Acc/train: 0.8524897042306252

epoch: 15, Loss/train: 0.4205549955368042, Acc/train: 0.8560464245600898

epoch: 16, Loss/train: 0.41469433903694153, Acc/train: 0.8589479595657057

epoch: 17, Loss/train: 0.40929552912712097, Acc/train: 0.8589479595657057

epoch: 18, Loss/train: 0.4043932557106018, Acc/train: 0.863253463122426

epoch: 19, Loss/train: 0.3995656669139862, Acc/train: 0.8626918757019842

epoch: 20, Loss/train: 0.3954036235809326, Acc/train: 0.8651254211905653

epoch: 21, Loss/train: 0.39138078689575195, Acc/train: 0.8647510295769375

epoch: 22, Loss/train: 0.38767680525779724, Acc/train: 0.8669037813552977

epoch: 23, Loss/train: 0.3841887414455414, Acc/train: 0.8678397603893673

epoch: 24, Loss/train: 0.3809354603290558, Acc/train: 0.8690565331336578

epoch: 25, Loss/train: 0.37777647376060486, Acc/train: 0.870928491201797

epoch: 26, Loss/train: 0.3752082884311676, Acc/train: 0.8696181205540996

epoch: 27, Loss/train: 0.3720647692680359, Acc/train: 0.8731748408835642

epoch: 28, Loss/train: 0.3695070445537567, Acc/train: 0.8741108199176338

epoch: 29, Loss/train: 0.36691951751708984, Acc/train: 0.8745788094346687

'''
