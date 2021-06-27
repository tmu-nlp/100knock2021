#knock81.py
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, padding_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size).to(device)
        
    def forward(self, x):
        self.batch_size = x.size()[0]
        embed = self.embedding(x)
        out, hidden = self.rnn(embed, self.init_hidden())
        out = F.relu(self.fc(out[:,-1,:]))
        return out