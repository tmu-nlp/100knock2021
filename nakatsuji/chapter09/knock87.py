#データ読み込み
train_df = pd.read_table(path + 'train.txt')
valid_df = pd.read_table(path + 'valid.txt')
test_df = pd.read_table(path + 'test.txt')

#データ抽出
X_train = train_df['TITLE']
X_valid = valid_df['TITLE']
X_test = test_df['TITLE']

dataset_train = CreateDataset(X_train, y_train, tokenizer, word2id)
dataset_valid = CreateDataset(X_valid, y_valid, tokenizer, word2id)
dataset_test = CreateDataset(X_test, y_test, tokenizer, word2id)

y_train = train_df['CATEGORY']
y_valid = valid_df['CATEGORY']
y_test = test_df['CATEGORY']

category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = y_train.map(lambda x: category_dict[x]).values
y_valid = y_valid.map(lambda x: category_dict[x]).values
y_test = y_test.map(lambda x: category_dict[x]).values


# 学習済みモデルを読み込む
model = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/basis2021/nlp100/GoogleNews-vectors-negative300.bin.gz', binary=True)


# 学習済み単語ベクトルを取得する
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
words_in_pretrained = 0
for i, word in enumerate(word2id.keys()):
    try:
        weights[i] = model[word]
        words_in_pretrained += 1
    except KeyError:
        weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
weights = torch.from_numpy(weights.astype((np.float32)))




# パラメータの設定
VOCAB_SIZE = len(set(word2id.values())) + 1
PADDING_IDX = len(set(word2id.values()))
BATCH_SIZE = 64
NUM_EPOCHS = 10

# モデル、損失関数、オプティマイザの定義
model = CNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4, emb_weights=weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

device = torch.device('cuda')

# モデルの学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)
#acc
_, acc_train = calculate_loss_and_accuracy(model, dataset_train, device=device, criterion=criterion)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test, device=device, criterion=criterion)
print(f'train acc: {acc_train}')
print(f'test acc: {acc_test}')