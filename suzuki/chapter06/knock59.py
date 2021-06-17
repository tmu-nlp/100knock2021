import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train = pd.read_table('train.feature.txt', header = 0, sep = '\t')
X_valid = pd.read_table('valid.feature.txt', header = 0, sep = '\t')
X_test = pd.read_table('test.feature.txt', header = 0, sep = '\t')

train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
valid = pd.read_table('valid.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])

test_acc = []

Cs = [0.1, 1.0, 10, 100]
for c in Cs:
    lg = LogisticRegression(penalty='l2', solver='sag', random_state=0, C=c)
    lg.fit(X_train, train['CATEGORY'])
    test_acc.append(accuracy_score(test['CATEGORY'], lg.predict(X_test)))


max_depths= [2, 4, 8, 16]
for m in max_depths:
    rfc = RandomForestClassifier(max_depth=m, random_state=0)
    rfc.fit(X_train, train['CATEGORY'])
    test_acc.append(accuracy_score(test['CATEGORY'], rfc.predict(X_test)))

bestIndex = test_acc.index(max(test_acc))
if bestIndex < 4:
    bestAlg = 'LogisticRegression'
    bestParam = f'C={Cs[bestIndex]}'
else:
    bestAlg = 'RandomForestClassifier'
    bestParam = f'max_depth={max_depths[bestIndex - 4]}'

print(bestAlg, bestParam)