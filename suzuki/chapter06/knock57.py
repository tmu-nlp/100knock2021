import knock52
import pandas as pd
import numpy as np

X_train = pd.read_table('train.feature.txt', header = 0, sep = '\t')

features = X_train.columns.values

for i in range(4):
  print('category: ' + knock52.lg.classes_[i])
  best = features[np.argsort(knock52.lg.coef_[0])[::-1][:10]] #lg.coefとX_trainの要素がリンクしてるからlgで並び替えたリストの要素番号をそのままX_trainに持ってこれる?
  worst = features[np.argsort(knock52.lg.coef_[0])[:10]]
  rank = 1
  print('best10')
  for j in best:
    print(str(rank) + ': ' + j)
    rank += 1
  rank = 1
  print('')
  print('worst10')
  for k in worst:
    print(str(rank) + ': ' + k)
    rank += 1
  print('')

'''

category: b
best10
1: fed
2: bank
3: ecb
4: ukraine
5: stocks
6: china
7: profit
8: oil
9: euro
10: update

worst10
1: and
2: the
3: her
4: video
5: ebola
6: study
7: aereo
8: she
9: star
10: facebook

category: e
best10
1: fed
2: bank
3: ecb
4: ukraine
5: stocks
6: china
7: profit
8: oil
9: euro
10: update

worst10
1: and
2: the
3: her
4: video
5: ebola
6: study
7: aereo
8: she
9: star
10: facebook

category: m
best10
1: fed
2: bank
3: ecb
4: ukraine
5: stocks
6: china
7: profit
8: oil
9: euro
10: update

worst10
1: and
2: the
3: her
4: video
5: ebola
6: study
7: aereo
8: she
9: star
10: facebook

category: t
best10
1: fed
2: bank
3: ecb
4: ukraine
5: stocks
6: china
7: profit
8: oil
9: euro
10: update

worst10
1: and
2: the
3: her
4: video
5: ebola
6: study
7: aereo
8: she
9: star
10: facebook

'''