# 71.単層ニューラルネットワークによる予測
# 問題70で保存した行列を読み込み，学習データについて以下の計算を実行せよ

import torch
import joblib
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    X_train = joblib.load('/users/kcnco/github/100knock2021/pan/chapter08/X_train.joblib')
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone() 
    # ネットワークを作成する
    torch.manual_seed(0) # シードを固定
    net = nn.Sequential(nn.Linear(300, 4), nn.Softmax(1))
    # 予測する
    y_pred = net(torch.reshape(X_train[0], (1, 300)))
    Y_pred = net(X_train[0:4])
    # 結果を表示する
    print(y_pred)
    print(Y_pred)

# 結果
# tensor([[0.2459, 0.2718, 0.2453, 0.2369]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2459, 0.2718, 0.2453, 0.2369],
#         [0.2472, 0.2497, 0.2619, 0.2411],
#         [0.2511, 0.2686, 0.2483, 0.2319],
#         [0.2460, 0.2647, 0.2562, 0.2330]], grad_fn=<SoftmaxBackward>)