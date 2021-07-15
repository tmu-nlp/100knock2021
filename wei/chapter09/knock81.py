""'''
81. RNNによる予測
RNNを用い，単語列xからカテゴリyを予測するモデルを実装
活性化関数gはtanh/ReLU
d_w,d_h = 300, 50
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset
from knock80 import get_word2id
import string


# データの準備
infile = '../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv'
df = pd.read_csv(infile, header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(
['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])


# ラベルベクトルを作成
category_dict = {'b':0, 't':1, 'e':2, 'm':3}
y_train = train['CATEGORY'].map(lambda x:category_dict[x]).values
y_valid = valid['CATEGORY'].map(lambda x:category_dict[x]).values
y_test = test['CATEGORY'].map(lambda x:category_dict[x]).values



# set parameter
word2id = get_word2id()
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  # h0のゼロベクトルを作成
        emb = self.emb(x)
        # emb.size() = (batch_size, seq_len, emb_size)
        out, hidden = self.rnn(emb, hidden)
        # out.size() = (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])
        # out.size() = (batch_size, output_size)
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden



def tokenizer(text, word2id=word2id, unk=0):
        # 記号をスペースに置換,スペースで分割したID列に変換(辞書になければunkで0を返す)
     res = []
     table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
     for word in text.translate(table).split():
        res.append(word2id.get(word, unk))
     return res


class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):     # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):   # Dataset[index]で返す値を指定
        text = self.X[index]
        inputs = self.tokenizer(text)

        return {
            'inputs': torch.tensor(inputs, dtype=torch.int64),
            'labels': torch.tensor(self.y[index], dtype=torch.int64)
        }


# Datesetを作成
#tokenizer = tokenizer(word2id)
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer)

if __name__ == '__main__':
    print(f'len(Dataset)の出力:{len(dataset_train)}')
    print(type(dataset_train))
    print('Dataset[index]の出力:')
    for var in dataset_train[1]:
        print(f'{var}: {dataset_train[1][var]}')
