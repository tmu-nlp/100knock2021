from knock52 import train_logistic
import pandas as pd

df = pd.read_csv(r'C:\Git\train.feature.txt', encoding='utf-8')
lr = train_logistic()
for labels in lr.classes_:
    for cat in lr.coef_:
        word_coef = [x for x in sorted(zip(cat,df.columns.tolist()))]
        print(labels)
        print(word_coef[:10])
        print(word_coef[-1:-11:-1])
