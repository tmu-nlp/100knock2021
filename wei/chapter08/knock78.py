'''
78. GPU上での学習
'''""
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import time
from knock70 import transform_w2v
from knock71 import SGLNet
from knock73 import NewsDataset

# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']),
    ['TITLE', 'CATEGORY']]

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


# GPUを指定し、75の損失計算関数を変えて、各データセットの損失を計算できる関数を定義
def calculate_loss_and_accuracy_GPU(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            # 入力TensorをGPUに送る
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


# knock77の関数train_modelを改変し、deviceを指定するための引数を追加
def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    # モデルをGPUに送る
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 開始時刻の記録
        s_time = time.time()

        # 訓練モードに設定
        model.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy_GPU(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_and_accuracy_GPU(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # チェックポイントの保存
        res_dir = './data/'
        model_param_dic = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_param_dic, res_dir + f'checkpoint_knock78_{epoch + 1}.pth')

        # 終了時刻の記録
        e_time = time.time()
        timeused = e_time - s_time

        # ログを出力
        print(
            f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, timeused : {timeused :.4f}sec')

    return {'train': log_train, 'valid': log_valid}


if __name__ == '__main__':

    device = torch.device('cuda')

    # モデルの定義
    SigleNNmodel = SGLNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(SigleNNmodel.parameters(), lr=1e-1)
    # モデルの学習
    for batch_size in [2 ** i for i in range(11)]:
        print(f'バッチサイズ: {batch_size}')
        log = train_model(dataset_train, dataset_valid, batch_size, SigleNNmodel, criterion, optimizer, 1,
                          device=device)
'''
バッチサイズ: 1
epoch: 1, loss_train: 0.3350, accuracy_train: 0.8842, loss_valid: 0.3550, accuracy_valid: 0.8855, timeused : 10.6196sec
バッチサイズ: 2
epoch: 1, loss_train: 0.3014, accuracy_train: 0.8968, loss_valid: 0.3305, accuracy_valid: 0.8922, timeused : 5.3526sec
バッチサイズ: 4
epoch: 1, loss_train: 0.2907, accuracy_train: 0.9017, loss_valid: 0.3258, accuracy_valid: 0.8952, timeused : 2.5744sec
バッチサイズ: 8
epoch: 1, loss_train: 0.2861, accuracy_train: 0.9029, loss_valid: 0.3222, accuracy_valid: 0.8937, timeused : 1.5297sec
バッチサイズ: 16
epoch: 1, loss_train: 0.2844, accuracy_train: 0.9029, loss_valid: 0.3208, accuracy_valid: 0.8967, timeused : 0.7759sec
バッチサイズ: 32
epoch: 1, loss_train: 0.2834, accuracy_train: 0.9034, loss_valid: 0.3202, accuracy_valid: 0.8960, timeused : 0.4495sec
バッチサイズ: 64
epoch: 1, loss_train: 0.2829, accuracy_train: 0.9038, loss_valid: 0.3199, accuracy_valid: 0.8967, timeused : 0.2879sec
バッチサイズ: 128
epoch: 1, loss_train: 0.2827, accuracy_train: 0.9038, loss_valid: 0.3197, accuracy_valid: 0.8967, timeused : 0.2221sec
バッチサイズ: 256
epoch: 1, loss_train: 0.2823, accuracy_train: 0.9037, loss_valid: 0.3197, accuracy_valid: 0.8967, timeused : 0.1735sec
バッチサイズ: 512
epoch: 1, loss_train: 0.2828, accuracy_train: 0.9037, loss_valid: 0.3196, accuracy_valid: 0.8967, timeused : 0.1906sec
バッチサイズ: 1024
epoch: 1, loss_train: 0.2805, accuracy_train: 0.9037, loss_valid: 0.3196, accuracy_valid: 0.8967, timeused : 0.2574sec
'''