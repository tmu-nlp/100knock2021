from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

train_path = './data/train.txt'
train_feature_path = './data/train.feature.txt'

#データ読み込み
names = ['TITLE', 'CATEGORY']
X_train = pd.read_csv(train_feature_path, sep='\t', header=None)
train_df = pd.read_csv(train_path, sep='\t', header=None, names=names)
Y_train = train_df['CATEGORY']

#学習
model = LogisticRegression(penalty="l2", solver="sag", random_state=0)
model.fit(X_train, Y_train)

#モデルの保存
joblib.dump(model, 'model.joblib')
