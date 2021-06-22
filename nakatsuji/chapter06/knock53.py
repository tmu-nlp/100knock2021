path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#データ読み込み
X_train = pd.read_csv(path + 'train_feature.csv', sep='\t')
Y_train = pd.read_table(path + 'train.txt')["CATEGORY"]
X_test = pd.read_csv(path + 'test_feature.csv', sep='\t')
Y_test = pd.read_table(path + 'test.txt')["CATEGORY"]

def score_lg(lg, X):
  return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

#モデル読み込み
filename = path + 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#予測
train_pred = score_lg(loaded_model, X_train)
test_pred = score_lg(loaded_model, X_test)