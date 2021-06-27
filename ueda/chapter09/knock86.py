#knock86.py
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, padding_idx, vocabweights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.conv = nn.Conv1d(embedding_size, hidden_size, 3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool1d(10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(self.dropout(x))
        return x

model = CNN(len(wordids)+1, 300, 50, 4, len(wordids), vocabweights)
model = model.to(device)
for inputs, target in train_loader:
    inputs = inputs.to(device)
    target = target.to(device)
    print(model(inputs))