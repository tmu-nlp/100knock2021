from gensim.models import KeyedVectors

load_model = KeyedVectors.load_word2vec_format(path + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

#データ準備
dataset_train = MyDataset(X_train, y_train, tokenizer, word2id)
dataset_valid = MyDataset(X_valid, y_valid, tokenizer, word2id)
dataset_test = MyDataset(X_test, y_test, tokenizer, word2id)

#パラメータ
vocab_size = len(set(word2id.values())) + 1
padding_idx = len(set(word2id.values()))
emb_dim, hidden_dim, output_dim = 300, 50, 4
learning_rate = 0.01
batch_size = 32
num_epochs = 10

device = torch.device('cuda')
weights = np.zeros((vocab_size, 300))
for i, word in enumerate(word2id.keys()):
    if not word in load_model.vocab:
        weights[i] = np.random.normal(scale=0.4, size=(300,))
    else:
        weights[i] = load_model[word]
weights = torch.from_numpy(weights.astype((np.float32)))    
model = MyRNN(vocab_size, emb_dim, hidden_dim, output_dim, padding_idx, weights)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

log = train_model(dataset_train, dataset_valid, batch_size, model, loss_fun, optimizer, \
                  num_epochs, Padsequence(padding_idx), device)

visualize_logs(log)
_, acc_train = calculate_loss_and_acc(model, dataset_train, device, loss_fun)
_, acc_test = calculate_loss_and_acc(model, dataset_test, device, loss_fun)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')