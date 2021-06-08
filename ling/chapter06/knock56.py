from sklearn.metrics import precision_score, recall_score, f1_score
from knock53 import train_pred,test_pred
from knock50_pd import train,test
import numpy as np
import pandas as pd
y_true=test['CATEGORY']
y_pred=test_pred[1]

precision=precision_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
recall = recall_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
f1=f1_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
print(precision)

micro_prec=precision_score(y_true, y_pred, average='micro')
macro_prec=precision_score(y_true, y_pred, average='macro')
micro_rec=recall_score(y_true, y_pred, average='micro')
macro_rec=recall_score(y_true, y_pred, average='macro')
micro_f1=f1_score(y_true, y_pred, average='micro')
macro_f1=f1_score(y_true, y_pred, average='macro')

precision=np.append(precision,[micro_prec,macro_prec])
recall=np.append(recall,[micro_rec,macro_rec])
f1=np.append(f1,[micro_f1,macro_f1])

scores = pd.DataFrame({'適合率': precision, '再現率': recall, 'F1スコア': f1},index=['b', 'e', 't', 'm', 'マイクロ平均', 'マクロ平均'])
print(scores)