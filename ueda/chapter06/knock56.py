from knock53 import pred_cat
from sklearn.metrics import classification_report
import pandas as pd

with open(r'C:\Git\valid.txt', encoding="utf-8") as f:
    df = pd.read_csv(r'C:\Git\valid.feature.txt', encoding='utf-8')
    pred, cat = pred_cat(df)
    print(classification_report([line.split('\t')[0] for line in f], cat))

