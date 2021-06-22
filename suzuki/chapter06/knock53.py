import numpy as np
import pandas as pd
import knock52

def score_lg(lg, X):
  return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
X_train = pd.read_table('train.feature.txt', header = 0, sep = '\t')
X_test = pd.read_table('test.feature.txt', header = 0, sep = '\t')

train_pred = score_lg(knock52.lg, X_train)
test_pred = score_lg(knock52.lg, X_test)

#knock54用
df_train_pred = pd.DataFrame(train_pred, index = ['pred', 'category']).T
df_test_pred = pd.DataFrame(test_pred, index = ['pred', 'category']).T

df_train_pred.to_csv('train.pred.txt', sep='\t', index=False)
df_test_pred.to_csv('test.pred.txt', sep='\t', index=False)

#確認
print('train_pred: {}'.format(train_pred))
print('test_pred: {}'.format(test_pred))
print('')
print(test)

'''

train_pred: [array([0.8025756 , 0.74973862, 0.89886111, ..., 0.90717206, 0.40460326,
       0.44426769]), array(['e', 'b', 'b', ..., 'e', 'b', 't'], dtype=object)]

test_pred: [array([0.4673741 , 0.83064716, 0.72640611, ..., 0.83082965, 0.94363889,
       0.95747415]), array(['b', 'b', 'e', ..., 'e', 'b', 'b'], dtype=object)]

     CATEGORY                                              TITLE
0           t  What Would Jobs Think of Getting in Bed With I...
1           b  European Car Sales Jump 7.6% as Price Cuts Hel...
2           b              Cynk Is a Joke, Not Proof of a Bubble
3           b  Gold Trades Above 3-Week Low Before Yellen's T...
4           e  Jamie Foxx Will Reportedly Play Mike Tyson In ...
...       ...                                                ...
1331        b    Slump in Erste Bank halts European stocks rally
1332        b  WATCH LIVE: Reuters Today - Publicis, Omnicom ...
1333        e  Gary Oldman Is Slammed By Mel Gibson's DUI Arr...
1334        b  TREASURIES-Prices inch lower in thin trading; ...
1335        b  FOREX-Dollar bounces on robust US payrolls dat...

[1336 rows x 2 columns]

'''