# 72.損失と勾配の計算
# 学習データの事例x1と事例集合x1,x2,x3,x4に対して，クロスエントロピー損失と，行列Wに対する勾配を計算せよ

import torch
import joblib
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    X_train = joblib.load('X_train.joblib')
    y_train = joblib.load('y_train.joblib')
    X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
    y_train = torch.from_numpy(y_train.astype(np.int64)).clone()
    # ネットワークを作成
    model = nn.Linear(300, 4)
    # 予測する
    x1 = torch.reshape(X_train[0], (1, 300))
    y_pred = torch.softmax(model.forward(x1), dim = -1)
    Y_pred = torch.softmax(model.forward(X_train[0:4]), dim = -1)
    loss_fn = nn.CrossEntropyLoss()
    # x1のクロスエントロピー損失と勾配を求める
    loss_y = loss_fn(y_pred, y_train[:1])
    model.zero_grad()
    loss_y.backward()
    print(f'x1のクロスエントロピー損失と勾配')
    print(f'クロスエントロピー損失: {loss_y}')
    print(f'勾配: {model.weight.grad}')
    # x1、x2、x3、x4のクロスエントロピー損失と勾配を求める
    loss_Y = loss_fn(Y_pred, y_train[0:4])
    model.zero_grad()
    loss_Y.backward()
    print(f'x1、x2、x3、x4のクロスエントロピー損失と勾配')
    print(f'クロスエントロピー損失: {loss_Y}')
    print(f'勾配: {model.weight.grad}')

# 結果
# x1のクロスエントロピー損失と勾配
# クロスエントロピー損失: 1.4462366104125977
# 勾配: tensor([[ 0.0031, -0.0053, -0.0025,  ..., -0.0016, -0.0239,  0.0214],
#         [-0.0003,  0.0004,  0.0002,  ...,  0.0001,  0.0020, -0.0018],
#         [-0.0007,  0.0012,  0.0006,  ...,  0.0004,  0.0054, -0.0048],
#         [-0.0021,  0.0037,  0.0017,  ...,  0.0011,  0.0165, -0.0148]])
#
# x1、x2、x3、x4のクロスエントロピー損失と勾配
# クロスエントロピー損失: 1.3693451881408691
# 勾配: tensor([[-8.0511e-04, -2.4086e-03, -1.9198e-03,  ..., -3.7956e-03,
#          -1.1013e-02,  7.0471e-03],
#         [-2.6161e-03,  2.6091e-04,  4.8492e-04,  ...,  2.9054e-03,
#           5.9318e-04, -6.6430e-04],
#         [ 2.8953e-03,  3.9552e-04,  5.9942e-04,  ..., -9.9664e-05,
#           4.8684e-03, -2.8080e-03],
#         [ 5.2585e-04,  1.7521e-03,  8.3544e-04,  ...,  9.8989e-04,
#           5.5511e-03, -3.5748e-03]])