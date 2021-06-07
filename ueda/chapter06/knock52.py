from numpy.core.fromnumeric import shape
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_logistic():
    df = pd.read_csv(r'C:\Git\train.feature.txt', encoding='utf-8')
    with open(r'C:\Git\train.txt', encoding="utf-8") as f:
        lr = LogisticRegression(random_state = 2021, max_iter=10000)
        lr.fit(df, [line.split('\t')[0] for line in f])
        return lr