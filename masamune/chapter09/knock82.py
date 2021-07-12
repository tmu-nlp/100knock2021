import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
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
    
class RNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, emb_size=300, output_size=4, hidden_size=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        emb = self.word_embedding(x)
        #(1, 300, 単語数)のテンソル
        hidden = self.init_hidden()
        out, hidden = self.rnn(emb, hidden)
        #out：(batch_size, seq_len, num_directions*hidden_size)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

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

    cat2num = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    y_train = train_df['CATEGORY'].map(lambda x: cat2num[x]).values
    y_valid = valid_df['CATEGORY'].map(lambda x: cat2num[x]).values
    y_test = test_df['CATEGORY'].map(lambda x: cat2num[x]).values

    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    dataset_train = CreatedDataset(train_df['TITLE'], y_train, give_id)
    dataset_valid = CreatedDataset(train_df['TITLE'], y_valid, give_id)
    dataset_test = CreatedDataset(test_df['TITLE'], y_valid, give_id)

    vocab_size = len(set(word2id.values())) + 1
    padding_idx = len(set(word2id.values())) #0埋め
    model = RNN(vocab_size=vocab_size, padding_idx=padding_idx)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    train(dataset_train, dataset_valid, model, criterion, optimizer)

    # 正解率を表示する
    _, acc_train = model.caluculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = model.caluculate_loss_and_accuracy(model, dataset_test)
    print(f'train accuracy: {acc_train}')
    print(f'test accuracy: {acc_test}')