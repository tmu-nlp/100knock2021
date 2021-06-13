from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import joblib

test_path = './data/test.txt'
test_feature_path = './data/test.feature.txt'

#データ読み込み
names = ['TITLE', 'CATEGORY']
X_test = pd.read_csv(test_feature_path, sep='\t', header=None)
test_df = pd.read_csv(test_path, sep='\t', header=None, names=names)
Y_test = test_df['CATEGORY']

#モデル読み込み
model = joblib.load('model.joblib')

#適合率、再現率、F!スコアの計算
Y_pred = model.predict(X_test)

precision = precision_score(Y_test, Y_pred, average=None)
precision_mi = precision_score(Y_test, Y_pred, average='micro')
precision_ma = precision_score(Y_test, Y_pred, average='macro')

recall = recall_score(Y_test, Y_pred, average=None)
recall_mi = recall_score(Y_test, Y_pred, average='micro')
recall_ma = recall_score(Y_test, Y_pred, average='macro')

f1 = f1_score(Y_test, Y_pred, average=None)
f1_mi = f1_score(Y_test, Y_pred, average='micro')
f1_ma = f1_score(Y_test, Y_pred, average='macro')

print(f'適合率: {precision}\tマイクロ平均: {precision_mi}\tマクロ平均: {precision_ma}')
print(f'再現率: {recall}\tマイクロ平均: {recall_mi}\tマクロ平均: {recall_ma}')
print(f'F1スコア: {f1}\tマイクロ平均: {f1_mi}\tマクロ平均: {f1_ma}')