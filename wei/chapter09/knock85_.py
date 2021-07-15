""'''
85.双方向RNN・多層化
双方向RNNを用いて、入力テキストをエンコードし、モデルを学習'''

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import torch
from sklearn.feature_extraction.text import CountVectorizer
from knock80_ import df2id
from knock82_ import list2tensor
from knock83_ import accuracy_gpu
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter()


base = '../chapter06/'
train = pd.read_csv(base + 'train.txt', header=None, sep='\t')
valid = pd.read_csv(base + 'valid.txt', header=None, sep='\t')
test = pd.read_csv(base + 'test.txt', header=None, sep='\t')

vectorizer = CountVectorizer(min_df=2)    #TFを計算。ただし、出現頻度が2回以上の単語だけを登録
train_title = train.iloc[:, 0].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()  #title corpusを入力とし、各文書を行に、.get_feature_names()を列に、TF array(スパース行列)を得る

sm = cnt.sum(axis=0)            # 列ごとに累加して、.get_feature_names()の単語ごとに、各docに出現頻度を数える
idx = np.argsort(sm)[::-1]      # 出現頻度の降順で、対応するindexを返す(.argsort返回数组值从小到大的对应索引值)
words = np.array(vectorizer.get_feature_names())[idx]   # ['w1',...,'wn'][index] indexで単語を索引し返す。最も出現した単語が先頭に
d = dict()
for i in range(len(words)):
    d[words[i]] = i + 1


max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1


class BiRNN(torch.nn.Module):
    def __init__(self):
        super(BiRNN, self).__init__()
        self.emb = torch.nn.Embedding(n_vocab, dw, padding_idx=PAD)
        self.rnn1 = torch.nn.RNN(dw, dh, bidirectional=True, batch_first=True)
        self.rnn2 = torch.nn.RNN(2*dh, dh, bidirectional=True,batch_first=True)
        self.rnn3 = torch.nn.RNN(2*dh, dh, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(2*dh, 4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn1(x, h)
        y, h = self.rnn2(y, h)
        y, h = self.rnn3(y, h)
        y = y[:, -1, :]
        y = self.linear(y)
        return y


X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)
X_train = list2tensor(X_train, max_len)
X_valid = list2tensor(X_valid, max_len)
X_test = list2tensor(X_test, max_len)

y_train = np.loadtxt('y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

# 事前学習済みの単語ベクトルで、emb.weightを初期化
w2v_model = KeyedVectors.load_word2vec_format('../chapter07/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# print(type(w2v_model))  ->class object

model = BiRNN()
with torch.no_grad():
    for k, v in d.items():
        if k in w2v_model.index_to_key:
            model.emb.weight[v] = torch.tensor(w2v_model[k], dtype=torch.float32)

    model.emb.weight = torch.nn.Parameter(model.emb.weight)
    # print(model.emb.weight)            ->何の操作もしなくて、実行ごとに出た値が違う　　？
    # print(model.emb.weight.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
# Dataloaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

if __name__ == '__main__':
    for i in tqdm(range(3), desc='Processing'):
        for epoch in range(10):
            epoch += 1
            for xx, yy in loader:
                y_pred = model(xx)
                loss = loss_func(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                y_pred = model(X_train.to(device))
                loss = loss_func(y_pred, y_train.to(device))
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('Accuracy/train', accuracy_gpu(y_pred, y_train), epoch)
                print(f'\nepoch:{epoch}')
                print(f'Accuracy on train:{accuracy_gpu(y_pred, y_train)}')


                y_pred = model(X_valid.to(device))
                loss = loss_func(y_pred, y_valid.to(device))
                writer.add_scalar('Loss/valid', loss, epoch)
                writer.add_scalar('Accuracy/valid', accuracy_gpu(y_pred, y_valid), epoch)
                print(f'Accuracy on valid:{accuracy_gpu(y_pred, y_valid)}')


'''
lr=1e-1
epoch:10
Accuracy on train:0.822163983526769
Accuracy on valid:0.8053892215568862
Processing: 100%|██████████| 3/3 [00:17<00:00,  5.84s/it]

lr=1e-3
epoch:10
Accuracy on train:0.5135716959940098
Accuracy on valid:0.5089820359281437
Processing: 100%|██████████| 3/3 [00:16<00:00,  5.49s/it]

'''