from knock53 import pred_cat
from sklearn.metrics import confusion_matrix
import pandas as pd

for data in ['train', 'valid']:
    with open(r'C:\Git\{}.txt'.format(data), encoding="utf-8") as f:
        df = pd.read_csv(r'C:\Git\{}.feature.txt'.format(data), encoding='utf-8')
        pred, cat = pred_cat(df)
        confusion = confusion_matrix([line.split('\t')[0] for line in f], cat, labels=['b', 'e', 'm', 't'])
        print(confusion)
