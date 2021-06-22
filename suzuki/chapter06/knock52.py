from sklearn.linear_model import LogisticRegression
import pandas as pd

train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
X_train = pd.read_table('train.feature.txt', header = 0, sep = '\t')

# モデルの学習
lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])

'''

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=10000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

'''