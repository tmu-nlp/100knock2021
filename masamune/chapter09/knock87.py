import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
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

def calculate_loss_and_accuracy(model, dataset, criterion=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs']
            labels = data['labels']
            outputs = model(inputs)

            # 損失計算
            if criterion != None:
                loss += criterion(outputs, labels).item()

            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(dataset), correct / total

def train(dataset_train, dataset_valid, model, criterion, optimizer, batch_size=50, num_epochs=10, collate_fn=None):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # 学習
    log_train, log_valid = [], []
    for epoch in range(num_epochs):

        model.train() # 訓練モードに設定

        for data in dataloader_train:
            optimizer.zero_grad() # 勾配をゼロで初期化

            inputs = data['inputs']
            labels = data['labels']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()# 評価モードに設定

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # ログを出力
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}') 

    return {'train': log_train, 'valid': log_valid}


if __name__ == '__main__':
    #データ読み込み
    names = ['TITLE', 'CATEGORY']
    train_df = pd.read_csv('../chapter06/data/train.txt', sep='\t', header=None, names=names)
    valid_df = pd.read_csv('../chapter06/data/valid.txt', sep='\t', header=None, names=names)
    test_df = pd.read_csv('../chapter06/data/test.txt', sep='\t', header=None, names=names)

    cat2num = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train_df['CATEGORY'].map(lambda x: cat2num[x]).values
    y_valid = valid_df['CATEGORY'].map(lambda x: cat2num[x]).values
    y_test = test_df['CATEGORY'].map(lambda x: cat2num[x]).values

    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    dataset_train = CreatedDataset(train_df['TITLE'], y_train, give_id)
    dataset_valid = CreatedDataset(train_df['TITLE'], y_valid, give_id)
    dataset_test = CreatedDataset(test_df['TITLE'], y_valid, give_id)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train(dataset_train, dataset_valid, model, criterion, optimizer)

    # 正解率を表示する
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train accuracy: {acc_train}')
    print(f'test accuracy: {acc_test}')