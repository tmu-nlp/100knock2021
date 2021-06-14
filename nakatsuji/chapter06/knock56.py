path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#データ読み込み
#X_train = pd.read_csv(path + 'train_feature.csv', sep='\t')
#Y_train = pd.read_table(path + 'train.txt')["CATEGORY"]
X_test = pd.read_csv(path + 'test_feature.csv', sep='\t')
Y_test = pd.read_table(path + 'test.txt')["CATEGORY"]

#モデル読み込み
filename = path + 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#予測
test_pred = loaded_model.predict(X_test)

#pre, rec, f1
pre = precision_score(Y_test, test_pred, average=None)
rec = recall_score(Y_test, test_pred, average=None)
f1 = recall_score(Y_test, test_pred, average=None)
#micro
micro_pre = precision_score(Y_test, test_pred, average='micro')
micro_rec = recall_score(Y_test, test_pred, average='micro')
micro_f1 = recall_score(Y_test, test_pred, average='micro')
#macro
macro_pre = precision_score(Y_test, test_pred, average='macro')
macro_rec = recall_score(Y_test, test_pred, average='macro')
macro_f1 = recall_score(Y_test, test_pred, average='macro')
##出力
print(f'適合率 : {pre} micro : {micro_pre} macro : {macro_pre}')
print(f'再現率 : {rec} micro : {micro_rec} macro : {macro_rec}')
print(f'F1スコア : {f1} micro : {micro_f1} macro : {macro_f1}')