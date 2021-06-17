from sklearn.metrics import accuracy_score
import pandas as pd

train_pred = pd.read_table('train.pred.txt', header = 0, sep = '\t', names = ['pred', 'category'])
test_pred = pd.read_table('test.pred.txt', header = 0, sep = '\t', names = ['pred', 'category'])
train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])

train_accuracy = accuracy_score(train['CATEGORY'], train_pred['category'])
test_accuracy = accuracy_score(test['CATEGORY'], test_pred['category'])
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')

'''

正解率（学習データ）：0.934
正解率（評価データ）：0.874

'''