import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

class CreatedDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_len):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.x[index]
        inputs = self.toknizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.y[index])
        }

class BERTclass(torch.nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.drop(out))
        return out

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuggle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for data in dataloader_train:
            ids = data['ids']
            mask = data['mask']
            labels = data['labels']
            optimizer.zero_grad()
            outputs = model.forward(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
        print(f'epoch: {epoch + 1}, loss_train: {loss_train}, accuracy_train: {acc_train}, loss_valid: {loss_valid}, accuracy_valid: {acc_valid}')    

def calculate_loss_and_accuracy(model, criterion, loader, device=None):
    model.eval()
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model.forward(ids, mask)
            # 損失を加算する
            loss += criterion(outputs, labels).item()
            # 正解数を数える
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

def calculate_accuracy(model, dataset, device=None):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model.forward(ids, mask)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    # Dataset読み込み
    df = pd.read_csv('../chapter06/data/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # one-hot化
    y_train = pd.get_dummies(train, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_valid = pd.get_dummies(valid, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_test = pd.get_dummies(test, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

    # Dataset作成
    max_len = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = CreatedDataset(train['TITLE'], y_train, tokenizer, max_len)
    dataset_valid = CreatedDataset(valid['TITLE'], y_valid, tokenizer, max_len)
    dataset_test = CreatedDataset(test['TITLE'], y_test, tokenizer, max_len)

    # ラベルを数値に変換する
    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
    y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
    y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

    # パラメータを設定する
    BATCH_SIZE = 50
    NUM_EPOCHS = 3

    # モデルを定義する
    model = BERTclass(output_size=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

    # 正解率を求める
    print(f'train acc: {calculate_accuracy(model, dataset_train)}')
    print(f'valid acc: {calculate_accuracy(model, dataset_valid)}')
    print(f'test  acc: {calculate_accuracy(model, dataset_test)}')