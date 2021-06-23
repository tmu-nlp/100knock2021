import pandas as pd
from gensim.models import KeyedVectors
import string
import torch

train = pd.read_table('train.txt', header = None, sep = '\t', names = ['category', 'title'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['category', 'title'])
valid = pd.read_table('valid.txt', header = None, sep = '\t', names = ['category', 'title'])

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

#特徴量ベクトルの作成
def w2v(sentence):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  words = sentence.translate(table).split() 
  vec = [model[word] for word in words if word in model]

  return torch.tensor(sum(vec) / len(vec))

X_train = torch.stack([w2v(text) for text in train['title']])
X_valid = torch.stack([w2v(text) for text in valid['title']])
X_test = torch.stack([w2v(text) for text in test['title']])

print(len(train))
print(X_train.size())
print(X_train)

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['category'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['category'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['category'].map(lambda x: category_dict[x]).values)

print(y_train.size())
print(y_train)

# 保存
torch.save(X_train, 'X_train.pt')
torch.save(X_valid, 'X_valid.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_train, 'y_train.pt')
torch.save(y_valid, 'y_valid.pt')
torch.save(y_test, 'y_test.pt')