'''
72. 損失と勾配の計算
compute CrossEntropyLoss, gradient of W 
'''""
import pandas as pd
import torch
from knock70 import transform_w2v
from knock71 import SGLNet
from torch import nn
from sklearn.model_selection import train_test_split

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

if __name__ == '__main__':
    # 単層ニューラルネットワークの初期化
    SigelNNmodel = SGLNet(300, 4)

    # 学習用のTensor型の平均化ベクトルとラベルベクトルを入力することで、集合にある各事例の平均損失を計算
    # 入力ベクトルはsoftmax前の値
    criterion = nn.CrossEntropyLoss()
    l_1 = criterion(SigelNNmodel(X_train[:1]), y_train[:1])
    SigelNNmodel.zero_grad()  # 勾配をゼロで初期化
    l_1.backward()  # 勾配を計算
    print(f'損失: {l_1:.4f}')
    print(f'勾配:\n{SigelNNmodel.fc.weight.grad}')

    l = criterion(SigelNNmodel(X_train[:4]), y_train[:4])
    SigelNNmodel.zero_grad()
    l.backward()
    print(f'損失: {l:.4f}')
    print(f'勾配:\n{SigelNNmodel.fc.weight.grad}')

'''
毎回実行すると、違った結果を得た
損失: 1.7513
勾配:
tensor([[-0.0691, -0.0047, -0.0056,  ..., -0.0621, -0.0358,  0.0717],
        [ 0.0056,  0.0004,  0.0005,  ...,  0.0050,  0.0029, -0.0058],
        [ 0.0546,  0.0037,  0.0045,  ...,  0.0491,  0.0283, -0.0567],
        [ 0.0089,  0.0006,  0.0007,  ...,  0.0080,  0.0046, -0.0093]])
損失: 1.4243
勾配:
tensor([[-0.0190,  0.0008,  0.0003,  ..., -0.0166, -0.0067,  0.0177],
        [-0.0019, -0.0019,  0.0141,  ...,  0.0140,  0.0302,  0.0097],
        [ 0.0310, -0.0040, -0.0306,  ..., -0.0059, -0.0052, -0.0091],
        [-0.0101,  0.0051,  0.0163,  ...,  0.0086, -0.0182, -0.0183]])'''

