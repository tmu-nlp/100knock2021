from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

train_path = './data/train.txt'
valid_path = './data/valid.txt'
test_path = './data/test.txt'
train_feature_path = './data/train.feature.txt'
valid_feature_path = './data/valid.feature.txt'
test_feature_path = './data/test.feature.txt'

#データ読み込み
names = ['TITLE', 'CATEGORY']
X_train = pd.read_csv(train_feature_path, sep='\t', header=None)
train_df = pd.read_csv(train_path, sep='\t', header=None, names=names)
X_valid = pd.read_csv(valid_feature_path, sep='\t', header=None)
valid_df = pd.read_csv(valid_path, sep='\t', header=None, names=names)
X_test = pd.read_csv(test_feature_path, sep='\t', header=None)
test_df = pd.read_csv(test_path, sep='\t', header=None, names=names)

result = []
for C in tqdm(np.logspace(-5, 4, 10, base=10)):
  # モデルの学習
  model = LogisticRegression(penalty='l2', solver='sag', random_state=0, C=C)
  model.fit(X_train, train_df['CATEGORY'])

  # 予測値の取得
  train_pred = model.predict(X_train)
  valid_pred = model.predict(X_valid)
  test_pred = model.predict(X_test)

  # 正解率の算出
  train_acc = accuracy_score(train_df['CATEGORY'], train_pred)
  valid_acc = accuracy_score(valid_df['CATEGORY'], valid_pred)
  test_acc = accuracy_score(test_df['CATEGORY'], test_pred)

  result.append([C, train_acc, valid_acc, test_acc])

#可視化
plt.plot(result[0], result[1], label="train")
plt.plot(result[0], result[2], label="valid")
plt.plot(result[0], result[3], label="test")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.show()