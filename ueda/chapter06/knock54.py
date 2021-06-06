from knock53 import pred_cat
from sklearn.metrics import accuracy_score
import pandas as pd

for data in ['train', 'test']:
    with open(r'C:\Git\{}.txt'.format(data), encoding="utf-8") as f:
        df = pd.read_csv(r'C:\Git\{}.feature.txt'.format(data), encoding='utf-8')
        pred, cat = pred_cat(df)
        accuracy = accuracy_score([line.split('\t')[0] for line in f], cat)
        print('正解率 = ' +str(accuracy))
