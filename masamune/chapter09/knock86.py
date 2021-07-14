import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import pickle

def give_id(sentence, word2id):
    words = sentence.split()
    ids = [word2id[word.lower()] for word in words]
    return ids

class CreatedDataset(Dataset):
    def __init__(self, x, y, give_id):
        self.x = x
        self.y = y
        self.give_id = give_id

    def __len__(self): # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index): # Dataset[index]で返す値を指定
        sentence = self.x[index]
        inputs = self.give_id(sentence=sentence, word2id=word2id)
        item = {}
        item['inputs'] = torch.tensor(inputs, dtype=torch.int64)
        item['labels'] = torch.tensor(self.y[index], dtype=torch.int64)
        return item

class CNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, emb_size=300, output_size=4, out_channels=100, kernel_heights=3, stride=1, padding=1, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        # x.size() = (batch_size, seq_len)
        emb = self.emb(x).unsqueeze(1) #emb.size() = (batch_size, 1, seq_len, emb_size)
        conv = self.conv(emb) # conv.size() = (batch_size, out_channels, seq_len, 1)
        act = F.relu(conv.squeeze(3)) #act.size() = (batch_size, out_channels, seq_len)
        max_pool = F.max_pool1d(act, act.size()[2]) # max_pool.size() = (batch_size, out_channels, 1) seq_len方向に最大値を取得
        out = self.fc(self.drop(max_pool.squeeze(2))) # out.size() = (batch_size, output_size)
        return out

if __name__ == '__main__':
    #データ読み込み
    names = ['TITLE', 'CATEGORY']
    train_df = pd.read_csv('../chapter06/data/train.txt', sep='\t', header=None, names=names)

    cat2num = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train_df['CATEGORY'].map(lambda x: cat2num[x]).values

    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    data_train = CreatedDataset(train_df['TITLE'], y_train, give_id)

    #学習済み単語ベクトル
    model = KeyedVectors.load_word2vec_format('../chapter07/GoogleNews-vectors-negative300.bin.gz', binary=True)

    vocab_size = len(set(word2id.values())) + 1
    padding_idx = len(set(word2id.values()))
    emb_size = 300
    weights = np.zeros((vocab_size, emb_size))
    words_in_pretrained = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(emb_size,))
    weights = torch.from_numpy(weights.astype((np.float32)))

    #モデル定義
    model = CNN(vacab_size=vocab_size, padding_idx=padding_idx)

    for i in range(5):
        x = data_train[i]['inputs']
        print(torch.softmax(model(x.unsqueeze(0)), dim=-1)) #入力毎にsoftmax計算