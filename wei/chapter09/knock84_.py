""'''
84. 単語ベクトルの導入
事前学習済みの単語ベクトルで単語埋め込みemb(x)を初期化し，学習せよ．'''
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import torch
from sklearn.feature_extraction.text import CountVectorizer
from knock80_ import df2id
from knock82_ import RNN,list2tensor
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
# print(len(words))     ->7570
d = dict()
for i in range(len(words)):
    d[words[i]] = i + 1


max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1


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

model = RNN()
# print(model.emb.weight)
'''when not using .no_grad() here, a RuntimeError was popped:
a view of a leaf Variable that requires grad is being used in an in-place operation.
need to learn more about this related'''
with torch.no_grad():
    for k, v in d.items():
        if k in w2v_model.index_to_key:
            model.emb.weight[v] = torch.tensor(w2v_model[k], dtype=torch.float32)


model.emb.weight = torch.nn.Parameter(model.emb.weight)
# print(model.emb.weight)
# print(model.emb.weight.shape)    -> torch.Size([7572, 300])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
# Dataloaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

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
lr = 1e-1
epoch:10
Accuracy on train:0.739797828528641
Accuracy on valid:0.7050898203592815
Processing: 100%|██████████| 3/3 [00:10<00:00,  3.42s/it]

lr = 1e-3
epoch:10
Accuracy on train:0.35819917633845
Accuracy on valid:0.3473053892215569

lr = 1e-5
epoch:10
Accuracy on train:0.29361662298764507
Accuracy on valid:0.3083832335329341
Processing: 100%|██████████| 3/3 [00:11<00:00,  3.89s/it]

-> lr = 0.1の方が精度が高いが、どうして？
'''