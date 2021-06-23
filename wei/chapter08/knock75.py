'''
75. 損失と正解率のプロット
'''""
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from  matplotlib import pyplot as plt
import numpy as np
from knock70 import transform_w2v
from knock71 import SGLNet
from knock73 import NewsDataset

# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']),
    ['TITLE','CATEGORY']]

# データの分割
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                               stratify=valid_test['CATEGORY'])
# 特徴ベクトルの作成
X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

# Datasetを作成するには、X_train, y_trainを利用
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

# Dataloaderの作成
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


# 各データセットの損失を計算できる関数を定義
def calculate_loss_and_accuracy(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total

if __name__ == '__main__':
    # モデルの定義
    SigleNNmodel = SGLNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(SigleNNmodel.parameters(), lr=1e-1)

    # 学習
    num_epochs = 30
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 訓練モードに設定
        SigleNNmodel.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = SigleNNmodel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 　損失と正解率を算出
        # 1エポック完了後、学習データと検証データ上の損失率と精度を計算
        loss_train, acc_train = calculate_loss_and_accuracy(SigleNNmodel, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_and_accuracy(SigleNNmodel, criterion, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # ログを出力
        print(
            f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')

    # 損失，正解率をグラフにプロットし，学習の進捗状況を確認
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log_train).T[0], label='train')
    ax[0].plot(np.array(log_valid).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log_train).T[1], label='train')
    ax[1].plot(np.array(log_valid).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.show()

