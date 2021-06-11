import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import *
import torch.nn as nn
import torch.nn.functional as F

train = pd.read_csv('../chapter06/data/train.txt', header=None, sep='\t')
valid = pd.read_csv('../chapter06/data/valid.txt', header=None, sep='\t')
test = pd.read_csv('../chapter06/data/test.txt', header=None, sep='\t') 

vectorizer = CountVectorizer(min_df=2)
train_title = train.iloc[:,1].str.lower()

cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()

for i in range(len(words)):
  d[words[i]] = i+1

def get_id(sentence):
    r = []
    for word in sentence:
        r.append(d.get(word,0))
    return r

def df2id(df):
    ids = []
    for i in df.iloc[:,1].str.lower():
        ids.append(get_id(i.split()))
    return ids

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
        seg_ids = torch.zeros_like(ids)
        attention_mask = (ids > 0)
        ret = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        last_hidden_state = ret["last_hidden_state"]
        x = last_hidden_state[:,0,:]
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
    pred = np.argmax(pred.data.numpy(), axis=1)
    label = label.data.numpy()
    return (pred == label).mean()

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt('../chapter08/data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('../chapter08/data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('../chapter08/data/y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = BertClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dfs_freeze(model)
model.fc.requires_grad_(True)

ds = TensorDataset(X_train.to(device), y_train.to(device))
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(30):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device))
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))

"""
epoch: 0
train loss: 1.1856974363327026, train acc: 0.47163983526769
valid loss: 1.2225379943847656, valid acc: 0.45434131736526945
epoch: 1
train loss: 1.146813154220581, train acc: 0.49185698240359416
valid loss: 1.159258484840393, valid acc: 0.4805389221556886
"""