#knock89.py
import transformers
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 20

def create_dataset(file_name):
    with open(file_name, encoding="utf-8") as f:
        x_vec=[]
        y_vec=[]
        for line in f:
            y, sent = line.strip().split('\t')
            x_vec.append(tokenizer.encode(sent, add_special_tokens=True))
            y_vec.append(int(y.translate(cat)))
        x_vec = ids2tensor(x_vec, max_len)
        y_vec = torch.tensor(y_vec, dtype = torch.int64)
        return x_vec, y_vec

class Bert(nn.Module):
    def __init__(self, output_size=4):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, output_size)
    
    def forward(self, ids):
        seg_ids = torch.zeros_like(ids)
        mask = (ids > 0)
        _, out = self.bert_model(input_ids = ids, token_type_ids = seg_ids, attention_mask=mask)
        out = self.fc(self.dropout(out))
        return out

device = torch.device('cuda')
model = Bert()
x_train, y_train = create_dataset(r'/content/drive/MyDrive/Dataset/train.txt')
x_test, y_test = create_dataset(r'/content/drive/MyDrive/Dataset/test.txt')
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)
optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()
train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []
for epoch in tqdm(range(5)):
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