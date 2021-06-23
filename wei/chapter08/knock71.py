'''
71.単層ニューラルネットワークによる予測
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from knock70 import transform_w2v
from torch import nn
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

X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])

# SGLNetという単層ニューラルネットワークを定義
class SGLNet(nn.Module):
    # 　ネットのlayerを定義
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    # 　forwardで入力データが順伝播時に通るレイヤーを順に配置しておく
    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 単層ニューラルネットワークの初期化
    SigelNNmodel = SGLNet(300, 4)
    # 未学習の行列Wで事例x_1を分類したとき，各カテゴリに属する確率を表すベクトル
    y_hat_1 = torch.softmax(SigelNNmodel(X_train[:1]), dim=-1)
    print(y_hat_1)

    Y_hat = torch.softmax(SigelNNmodel.forward(X_train[:4]), dim=-1)
    print(Y_hat)
