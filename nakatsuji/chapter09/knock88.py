#85でのbiRNNをより多層化する
#パラメータ
vocab_size = len(set(word2id.values())) + 1
padding_idx = len(set(word2id.values()))
emb_dim, hidden_dim, output_dim = 300, 50, 4
num_layers = 5

learning_rate = 0.01
batch_size = 32
num_epochs = 10

device = torch.device('cuda')

model = MyRNN(vocab_size, emb_dim, hidden_dim, output_dim, num_layers, padding_idx, \
              bidirectional=True)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

log = train_model(dataset_train, dataset_valid, batch_size, model, loss_fun, optimizer, \
                  num_epochs, Padsequence(padding_idx), device)