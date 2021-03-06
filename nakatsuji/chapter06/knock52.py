path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv(path + 'train_feature.csv', sep='\t')
Y_train = pd.read_table(path + 'train.txt')['CATEGORY']
Y_train.shape

#モデルの学習
model = LogisticRegression()
model.fit(X_train, Y_train)

#モデルの保存
filename = path + 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))