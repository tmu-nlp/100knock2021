""'''
[task description]正解率の計測
52で学習したロジスティック回帰モデルの正解率を，
学習データおよび評価データ上で計測せよ
'''

from sklearn.metrics import accuracy_score
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, train_seg, test_seg
from knock53 import score_lg
from sklearn.linear_model import LogisticRegression

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

    lgfitT = lg.fit(X_train, train['CATEGORY'])
    train_pred = score_lg(lgfitT, X_train)
    train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
    print(f'正解率(学習データ)　:　{train_accuracy:.3f}')

    lgfitTe = lg.fit(X_test, test['CATEGORY'])
    test_pred = score_lg(lgfitTe, X_test)
    test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])
    print(f'正解率(評価データ)　:　{test_accuracy:.3f}')

'''
正解率(学習データ)　:　0.927
正解率(評価データ)　:　0.894
'''