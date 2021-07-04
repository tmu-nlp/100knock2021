from torch.nn.utils.rnn import pad_sequence
#ミニバッチ化するには系列長を揃える必要あり
class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
    
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])

        return {'inputs' : sequences_padded , 'labels' : labels}

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

model = MyRNN(vocab_size, emb_dim, hidden_dim, output_dim)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

log = train_model(dataset_train, dataset_valid, batch_size, model, loss_fun, optimizer, \
                  num_epochs, Padsequence(padding_idx), device)


visualize_logs(log)
_, acc_train = calculate_loss_and_acc(model, dataset_train, device, loss_fun)
_, acc_test = calculate_loss_and_acc(model, dataset_test, device, loss_fun)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')