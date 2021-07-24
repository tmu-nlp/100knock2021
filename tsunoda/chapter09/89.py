from transformers import 

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def read_for_bert(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_t = [categories.index(line[0]) for line in dataset]
    dataset_x = [torch.tensor(tokenizer.encode(line[1]), dtype=torch.long) for line in dataset]
    return dataset_x, torch.tensor(dataset_t, dtype=torch.long)

bert_train_x, bert_train_t = read_for_bert('data/train.txt')
bert_valid_x, bert_valid_t = read_for_bert('data/valid.txt')
bert_test_x, bert_test_t = read_for_bert('data/test.txt')

class BertDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        src = [torch.cat([x['src'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        src = torch.stack(src)
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':src,
            'trg':torch.tensor([x['trg'] for x in xs]),
            'mask':mask,
        }

bert_train_dataset = BertDataset(bert_train_x, bert_train_t)
bert_valid_dataset = BertDataset(bert_valid_x, bert_valid_t)
bert_test_dataset = BertDataset(bert_test_x, bert_test_t)

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-cased', num_labels=4)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-cased', config=config)

    def forward(self, batch):
        x = self.bert(batch['src'], attention_mask=batch['mask'])
       
       
model = BertClassifier()
loaders = (
    gen_maxtokens_loader(bert_train_dataset, 1000),
    gen_descending_loader(bert_valid_dataset, 32),
)
task = Task()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
trainer = Trainer(model, loaders, task, optimizer, 5, device)
trainer.train()

predictor = Predictor(model, gen_loader(bert_train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))

predictor = Predictor(model, gen_loader(bert_test_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(test_t, pred))
