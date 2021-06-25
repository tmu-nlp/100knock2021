import torch
import numpy as np

#単層ニューラルネット
class NN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, output_size, bias=False) #全結合層の定義
        torch.nn.init.normal_(self.layer1.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, input):
        activation = torch.nn.Softmax(dim=-1)
        output = activation(self.layer1(input))

        return output

#モデル読み込み
model = NN(300, 4)
model.load_state_dict(torch.load('./model.pth'))

#データ読み込み
X_train = np.loadtxt('./data/X_train.txt', delimiter=',')
X_train = torch.tensor(X_train, dtype=torch.float32) 
y_train = np.loadtxt('./data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.long)
X_valid = np.loadtxt('./data/X_valid.txt', delimiter=',')
X_valid = torch.tensor(X_valid, dtype=torch.float32) 
y_valid = np.loadtxt('./data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.long)

#評価関数
def accuracy(probs, y):
    cnt = 0
    for i, prob in enumerate(probs):
        #tensorからndarrayに変換し、最大要素のindexを返す
        y_pred = np.argmax(prob.detach().numpy())
        if y_pred == y.detach().numpy()[i]:
            cnt += 1
    
    return cnt/len(y)

probs = model(X_train)
print(f'accuracy (train)：{accuracy(probs, y_train)}')
probs = model(X_valid)
print(f'accuracy (valid)：{accuracy(probs, y_valid)}')

'''
accuracy (train)：0.8843703148425787
accuracy (valid)：0.8793103448275862
'''