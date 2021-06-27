#knock84.py
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, padding_idx, vocabweights):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(vocabweights, padding_idx=padding_idx)
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

wordvec = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Git\GoogleNews-vectors-negative300.bin.gz', binary=True)
vocabweights = np.zeros((len(wordids)+1, 300))
for i, word in enumerate(wordids.keys()):
    if word in wordvec:
        vocabweights[i] = wordvec[word]
    else:
        vocabweights[i] = 0.2*np.random.rand(300)-0.1
vocabweights = torch.from_numpy(vocabweights.astype((np.float32)))

device = torch.device('cuda')
model = RNN(len(wordids)+1, 300, 50, 4, len(wordids), vocabweights)
x_train, y_train = create_dataset(r'C:\Git\train.txt')
x_test, y_test = create_dataset(r'C:\Git\test.txt')
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, batch_size = 16, shuffle = True)
optim = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []
for epoch in tqdm(range(10)):
    for inputs, target in train_loader:
        optim.zero_grad()
        model.to(device)
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optim.step()
    with torch.no_grad():
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        pred = model(x_train)
        train_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_train))
        train_loss.append(loss_func(pred, y_train.to(device)).detach().cpu().numpy())
        pred = model(x_test)
        test_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_test))
        test_loss.append(loss_func(pred, y_test.to(device)).detach().cpu().numpy())

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.plot(train_accuracy, label='train')
plt.plot(test_accuracy, label='test')
plt.subplot(1,2,2)
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.show()