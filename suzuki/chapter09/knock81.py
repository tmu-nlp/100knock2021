import torch

dw = 300
dh = 50
words_length = 7612

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(words_length + 1,dw)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        y = self.softmax(y)
        return y