from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from knock53 import train_pred,test_pred
from knock50_pd import train,test

'''
confusion_matrix(y_true, y_pred)
@para
y_true: label
y_pred: labels predicted by model
'''

#学習データの混同行列
train_cm = confusion_matrix(train['CATEGORY'], train_pred[1])
print('学習データの混同行列：\n')
print(train_cm)
#評価データの混同行列
test_cm = confusion_matrix(test['CATEGORY'], test_pred[1])
print('評価データの混同行列：\n')
print(test_cm)


#venv of sklearn: "/opt/homebrew/Caskroom/miniforge/base/envs/sk-env"