from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

train_path = './data/train.txt'
test_path = './data/test.txt'
train_feature_path = './data/train.feature.txt'
test_feature_path = './data/test.feature.txt'

#データ読み込み
names = ['TITLE', 'CATEGORY']
X_train = pd.read_csv(train_feature_path, sep='\t', header=None)
train_df = pd.read_csv(train_path, sep='\t', header=None, names=names)
X_test = pd.read_csv(test_feature_path, sep='\t', header=None)
test_df = pd.read_csv(test_path, sep='\t', header=None, names=names)

#モデル読み込み
model = joblib.load('model.joblib')

#正解率
train_acc = accuracy_score(train_df['CATEGORY'], model.predict(X_train))
test_acc = accuracy_score(test_df['CATEGORY'], model.predict(X_test))

print(f'train_acc: {train_acc}')
print(f'test_acc: {test_acc}')