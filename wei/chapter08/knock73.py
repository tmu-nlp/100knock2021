from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from knock70 import transform_w2v
from knock71 import SGLNet

# データの読込
df = pd.read_csv('./../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv', header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[
    df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE',
                                                                                                             'CATEGORY']]

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


# 学習用のTensor型の平均化ベクトルとラベルベクトルを変換
class NewsDataset(Dataset):
    def __init__(self, X, y):  # datasetの構成要素を指定
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]で返す値を指定
        return [self.X[idx], self.y[idx]]


# Datasetを作成するには、X_train, y_trainを利用
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

# Dataloaderの作成
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

'''データの準備ができて、行列Wを学習
まず、前問と同様に、単層モデル、損失関数を定義
計算した勾配から重みを更新するには、optimizerをSGDに指定に定義
エポック数を10にして学習を実行
'''

if __name__ == '__main__':
    # モデルの定義
    SigleNNmodel = SGLNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(SigleNNmodel.parameters(), lr=1e-1)

    # 学習
    num_epochs = 10
    for epoch in range(num_epochs):
        # 訓練モードに設定
        SigleNNmodel.train()
        loss_train = 0.0
        for i, (inputs, labels) in enumerate(dataloader_train):
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = SigleNNmodel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 損失を記録
            loss_train += loss.item()

        # バッチ単位の平均損失計算
        loss_train = loss_train / i

        # 検証データの損失計算
        SigleNNmodel.eval()
        with torch.no_grad():
            inputs, labels = next(iter(dataloader_valid))
            outputs = SigleNNmodel(inputs)
            loss_valid = criterion(outputs, labels)

        # ログを出力
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')

'''
epoch: 1, loss_train: 0.4492, loss_valid: 0.3622
epoch: 2, loss_train: 0.3069, loss_valid: 0.3304
epoch: 3, loss_train: 0.2790, loss_valid: 0.3150
epoch: 4, loss_train: 0.2651, loss_valid: 0.3092
epoch: 5, loss_train: 0.2553, loss_valid: 0.3068
epoch: 6, loss_train: 0.2477, loss_valid: 0.3107
epoch: 7, loss_train: 0.2424, loss_valid: 0.3056
epoch: 8, loss_train: 0.2385, loss_valid: 0.3033
epoch: 9, loss_train: 0.2349, loss_valid: 0.3040
epoch: 10, loss_train: 0.2328, loss_valid: 0.3062
'''
