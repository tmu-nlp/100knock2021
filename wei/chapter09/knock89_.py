""'''
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，
ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import cuda
import time
from matplotlib import pyplot as plt
import transformers
from transformers import BertTokenizer, BertModel


infile = '../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv'
df = pd.read_csv(infile,header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]

train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid ,test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# print(train.head())

# Datasetを定義
class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.y[index])
        }

# 正解ラベルのone-hotベクトル
y_train = pd.get_dummies(train, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
y_valid = pd.get_dummies(valid, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
y_test = pd.get_dummies(test, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

# Datasetの作成,文字列をidsというID系列に変換
max_len = 20
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   # 小文字のみ
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, max_len)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, max_len)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, max_len)

for var in dataset_train[0]:
    print(f'{var}:{dataset_train[0][var]}')

'''
ids:tensor([  101, 25416,  9463,  1011, 10651,  1015,  1011,  2647,  2482,  4341,
         2039,  2005,  4369,  3204,  2004, 18730,  8980,   102,     0,     0])
mask:tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
labels:tensor([1., 0., 0., 0.])
変換の過程で、入力文の文頭と文末に[CLS]、[SEP]がそれぞれ挿入されるため、それぞれ101、102に対応
0はパでイング
one-hotベクトルを用いて、正解ラベルをlabelsに保持
'''
# bert分類モデルの定義
class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, output_size):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out_temp = self.fc(self.drop(out))
        out = torch.tensor(out_temp)
        return out


# 損失と正解率を計算
def calculate_loss_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # デバイスを指定
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            # 順伝播
            outputs = model(ids, mask)

            # 損失計算
            loss += criterion(outputs, labels).item()
            # 正解率計算
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

# モデルの学習を実行し、損失・正解率のロゴを返す
def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time()
        model.train()
        for data in dataloader_train:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            # 勾配をゼロで初期化
            optimizer.zero_grad()

            outputs = model.forward(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward
            optimizer.step()

            # 損失と正解率を算出
            loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train, device)
            loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid, device)
            log_train.append([log_train, acc_train])
            log_valid.append([loss_valid, acc_valid])

            # チェックポイントを保存
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()},f'checkpoint{epoch + 1}.pt')

            e_time = time.time()
            time_used = e_time - s_time
            print(f'epoch: {epoch +1 }, loss_trian:{loss_train:.4f}, accuracy_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, accuracy_valid:{acc_valid:.4f}, time_used:{time_used:.4f}sec')
        return {'train': log_train, 'valid':log_valid}


# パラメータを設定
DROP_RATE = 0.4
OUTPUT_SIZE = 4
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

# モデル、損失関数、optimizerを定義し、デバイスを指定して、モデルを学習
model = BERTClass(DROP_RATE, OUTPUT_SIZE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
device = 'cuda' if cuda.is_available() else 'cpu'

log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device=device)





