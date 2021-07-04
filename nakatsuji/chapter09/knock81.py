from torch.autograd import Variable
class MyDataset(Dataset):
    def __init__(self, X, y, tokenizer, word2id):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.tokenizer(text=text)
        return {'inputs': torch.tensor(inputs, dtype=torch.int64), 
                'labels': torch.tensor(self.y[index], dtype=torch.int64)
                }


class MyRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, num_layers, padding_idx, emb_weights=None, bidirectional=False):
        super(MyRNN, self).__init__()
        self.device = torch.device('cuda')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim,  num_layers, nonlinearity='tanh', bidirectional=bidirectional,batch_first=True)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.fc = nn.Softmax()
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device)
        return hidden

    def forward(self, x):
        self.batch_size = x.size()[0]
        emb_x = self.emb(x)
        h0 = self.init_hidden()
        out, hn = self.rnn(emb_x, h0)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.fc(out)
        return out
