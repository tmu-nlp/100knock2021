from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_pred = pd.read_table('train.pred.txt', header = 0, sep = '\t', names = ['pred', 'category'])
test_pred = pd.read_table('test.pred.txt', header = 0, sep = '\t', names = ['pred', 'category'])
train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])

# 学習データ
train_cm = confusion_matrix(train['CATEGORY'], train_pred['category'])
print('学習データ')
print(train_cm)
sns.heatmap(train_cm, annot=True, cmap='Blues')
plt.show()

# 評価データ
test_cm = confusion_matrix(test['CATEGORY'], test_pred['category'])
print('評価データ')
print(test_cm)
sns.heatmap(test_cm, annot=True, cmap='Blues')
plt.show()

'''
学習データ
[[4423   65    6   42]
 [  38 4165    1    9]
 [ 100  115  505    9]
 [ 200  113    6  887]]

評価データ
[[519  21   0   8]
 [ 17 526   1   3]
 [ 23  15  41   2]
 [ 50  25   3  82]]

'''