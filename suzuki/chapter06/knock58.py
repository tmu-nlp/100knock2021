from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knock53 import score_lg


X_train = pd.read_table('train.feature.txt', header = 0, sep = '\t')
X_valid = pd.read_table('valid.feature.txt', header = 0, sep = '\t')
X_test = pd.read_table('test.feature.txt', header = 0, sep = '\t')

train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
valid = pd.read_table('valid.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])

result = []
for C in tqdm(np.logspace(-5, 4, 10, base=10)):
  # モデルの学習
  lg = LogisticRegression(random_state=123, max_iter=10000, C=C)
  lg.fit(X_train, train['CATEGORY'])

  # 予測値の取得
  train_pred = score_lg(lg, X_train)
  valid_pred = score_lg(lg, X_valid)
  test_pred = score_lg(lg, X_test)

  # 正解率の算出
  train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
  valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])
  test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])

  # 結果の格納
  result.append([C, train_accuracy, valid_accuracy, test_accuracy])


# 視覚化
result = np.array(result).T
plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='valid')
plt.plot(result[0], result[3], label='test')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xscale ('log')
plt.xlabel('C')
plt.legend()
plt.show()