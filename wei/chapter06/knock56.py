""'''
[task description]適合率，再現率，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，
評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．
'''

from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg
from knock53 import score_lg
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_scores(y_true, y_pred):
    # 適合率
    precision = precision_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
    precision = np.append(precision, precision_score(y_true, y_pred, average='micro'))
    precision = np.append(precision, precision_score(y_true, y_pred, average='macro'))

    # 再現率
    recall = recall_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
    recall = np.append(recall, recall_score(y_true, y_pred, average='micro'))
    recall = np.append(recall, recall_score(y_true, y_pred, average='macro'))

    # F1スコア
    f1 = f1_score(test['CATEGORY'], test_pred[1], average=None, labels=['b', 'e', 't', 'm'])
    f1 = np.append(f1, f1_score(y_true, y_pred, average='micro'))
    f1 = np.append(f1, f1_score(y_true, y_pred, average='macro'))

    # 結果を結合してデータフレーム化
    scores = pd.DataFrame({'適合率': precision, '再現率': recall, 'F1スコア': f1},
                          index=['b', 'e', 't', 'm', 'マイクロ平均', 'マクロ平均'])

    return scores



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

    lgfitTe = lg.fit(X_test, test['CATEGORY'])
    test_pred = score_lg(lgfitTe, X_test)
    print(calculate_scores(test['CATEGORY'], test_pred[1]))

'''
             適合率       再現率     F1スコア
b       0.881141  0.987567  0.931323
e       0.888702  0.994340  0.938557
t       0.987179  0.506579  0.669565
m       1.000000  0.373626  0.544000
マイクロ平均  0.893713  0.893713  0.893713
マクロ平均   0.939256  0.715528  0.770861
'''