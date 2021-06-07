""'''
[task description]
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
'''

from sklearn.linear_model import LogisticRegression
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, train_seg



# def lg(train_feature, train_categ):
#     lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')
#     return lg.fit(train_feature,train_categ)

if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = train_seg(df_pre, train, valid)
    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')
    lgfit = lg.fit(X_train, train['CATEGORY'])
    print(lgfit)

'''
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=10000, multi_class='auto',
          n_jobs=None, penalty='l2', random_state=123, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
          '''
