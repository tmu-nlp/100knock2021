import gensim
import gzip
import torch
import pandas as pd

model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Git\GoogleNews-vectors-negative300.bin.gz', binary=True)

def vector_xy(data_name):
    x_vec = []
    y_vec = []
    with open(data_name, encoding="utf-8") as f:
        for line in f:
            y, x = line.strip().split("\t")
            for word in x.split(" "):
                if word not in model: continue
                x_vec.append(model[word])
            y_vec.append(y)
    y_vec = pd.factorize(y_vec, sort=True)[0]
    x_vec = torch.tensor(x_vec)
    y_vec = torch.tensor(y_vec)
    return x_vec, y_vec

x_train, y_train = vector_xy(r'C:\Git\train.txt')
x_valid, y_valid = vector_xy(r'C:\Git\valid.txt')
x_test, y_test = vector_xy(r'C:\Git\test.txt')
torch.save(x_train, r'C:\Git\x_train.pt')
torch.save(y_train, r'C:\Git\y_train.pt')
torch.save(x_valid, r'C:\Git\x_valid.pt')
torch.save(y_valid, r'C:\Git\y_valid.pt')
torch.save(x_test, r'C:\Git\x_test.pt')
torch.save(y_test, r'C:\Git\y_test.pt')