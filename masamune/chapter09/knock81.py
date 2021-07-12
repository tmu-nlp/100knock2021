import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import pickle

def give_id(sentence, word2id):
    words = sentence.split()
    ids = [word2id[word.lower()] for word in words]
    return ids

class CreateDataset(Dataset):
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

if __name__ == '__main__':
    #データ読み込み
    names = ['TITLE', 'CATEGORY']
    train_df = pd.read_csv('../chapter06/data/train.txt', sep='\t', header=None, names=names)

    cat2num = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train_df['CATEGORY'].map(lambda x: cat2num[x]).values

    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    data_train = CreateDataset(train_df['TITLE'], y_train, give_id)

    vocab_size = len(set(word2id.values())) + 1
    padding_idx = len(set(word2id.values())) #0埋め

    model = RNN(vocab_size=vocab_size, padding_idx=padding_idx)

    for i in range(5):
        x = data_train[i]['inputs']
        print(torch.softmax(model(x.unsqueeze(0)), dim=-1)) #入力毎にsoftmax計算

'''
tensor([[0.3182, 0.2520, 0.1494, 0.2804]], grad_fn=<SoftmaxBackward>)
tensor([[0.2176, 0.1921, 0.2524, 0.3379]], grad_fn=<SoftmaxBackward>)
tensor([[0.3064, 0.2748, 0.2605, 0.1583]], grad_fn=<SoftmaxBackward>)
tensor([[0.3382, 0.2827, 0.2416, 0.1375]], grad_fn=<SoftmaxBackward>)
tensor([[0.2356, 0.2925, 0.2820, 0.1899]], grad_fn=<SoftmaxBackward>)
'''