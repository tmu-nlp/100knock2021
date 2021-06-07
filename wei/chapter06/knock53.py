""'''
[task description]
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
'''

import numpy as np
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, train_seg, test_seg
# from knock52 import lg
from sklearn.linear_model import LogisticRegression

def score_lg(lg, X):
    return [np.max(lg.fit.predict_proba(X), axis=1), lg.fit.predict(X)]


if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = train_seg(df_pre, train, valid)
    X_test = test_seg(df_pre, train, valid)
    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')
    # lg = lg(X_train, train['CATEGORY'])
    train_pred = score_lg(lg, X_train)
    test_pred = score_lg(lg, X_test)
    print(train_pred)