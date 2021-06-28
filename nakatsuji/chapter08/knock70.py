path = "/content/drive/MyDrive/basis2021/nlp100/"
import pandas as pd
import gensim
train_df = pd.read_table(path + 'train.txt')
valid_df = pd.read_table(path + 'valid.txt')
test_df = pd.read_table(path + 'test.txt')

file =path + 'GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

import string
import torch

def transform_word2vec(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = text.translate(table).split()  
    vec = [model[word] for word in words if word in model]  
    return torch.tensor(sum(vec) / len(vec)) 
X_train = torch.stack([transform_word2vec(text) for text in train_df['TITLE']])
X_valid = torch.stack([transform_word2vec(text) for text in valid_df['TITLE']])
X_test = torch.stack([transform_word2vec(text) for text in test_df['TITLE']])

category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train_df['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid_df['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test_df['CATEGORY'].map(lambda x: category_dict[x]).values)

torch.save(X_train, path + 'X_train.pt')
torch.save(X_valid, path + 'X_valid.pt')
torch.save(X_test, path + 'X_test.pt')
torch.save(y_train, path + 'y_train.pt')
torch.save(y_valid, path + 'y_valid.pt')
torch.save(y_test, path + 'y_test.pt')