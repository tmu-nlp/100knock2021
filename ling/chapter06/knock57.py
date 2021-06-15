from sklearn.metrics import precision_score, recall_score, f1_score
from knock53 import train_pred,test_pred
from knock50_pd import train,test
from knock52 import LR
from knock51_pd import X_train
import numpy as np
import pandas as pd


'''
LR.classes_: A list of class labels known to the classifier
LR.coef_: Coefficient of the features in the decision function.
'''

features= X_train.columns.values
index = [i for i in range(1, 11)]
for c, coef in zip(LR.classes_, LR.coef_):
  print(f'【カテゴリ】{c}')
  best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
  worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
  print(pd.concat([best10, worst10], axis=0))
  print('\n')
#print(LR.classes_)
#print(LR.coef_)