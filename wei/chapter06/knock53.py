""'''
[task description]予測
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
'''

import numpy as np
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg
from sklearn.linear_model import LogisticRegression



def score_lg(lg, X):
    return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]


if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = seg(df_pre, train, valid)[0]
    X_test = seg(df_pre, train, valid)[2]
    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')

    lgfitT = lg.fit(X_train, train['CATEGORY'])
    train_pred = score_lg(lgfitT, X_train)
    print(train_pred)

    lgfitTe = lg.fit(X_test, test['CATEGORY'])
    test_pred = score_lg(lgfitTe, X_test)
    print(test_pred)

'''
train_pred
[array([0.84030117, 0.67899853, 0.55636978, ..., 0.86051001, 0.61358001,
       0.90829256]), array(['b', 't', 'm', ..., 'b', 'm', 'e'], dtype=object)]
test_pred
[array([0.68223717, 0.65320246, 0.85278413, ..., 0.87394799, 0.79795111,
       0.39329039]), array(['e', 'e', 'b', ..., 'e', 'e', 't'], dtype=object)]
       '''