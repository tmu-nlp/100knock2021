path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

#データ読み込み
X_train = pd.read_csv(path + 'train_feature.csv', sep='\t')
Y_train = pd.read_table(path + 'train.txt')["CATEGORY"]
X_test = pd.read_csv(path + 'test_feature.csv', sep='\t')
Y_test = pd.read_table(path + 'test.txt')["CATEGORY"]

#モデル読み込み
filename = path + 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#混同行列
cm_train = confusion_matrix(Y_train, loaded_model.predict(X_train))
cm_test = confusion_matrix(Y_test, loaded_model.predict(X_test))

print('train confusion matrix')
print(cm_train)
print('test confusion matrix')
print(cm_test)