""'''
[task description]学習
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
'''

from sklearn.linear_model import LogisticRegression
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg



if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = seg(df_pre, train, valid)[0]
    # モデルの学習
    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')
    lgfit = lg.fit(X_train, train['CATEGORY'])
    print(lgfit)

'''
LogisticRegression(max_iter=10000, random_state=123)'''
