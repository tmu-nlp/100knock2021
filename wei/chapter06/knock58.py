'''
[task description]正則化パラメータの変更
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
'''

from sklearn.metrics import accuracy_score
from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg
from knock53 import score_lg
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = seg(df_pre, train, valid)[0]
    X_valid = seg(df_pre, train, valid)[1]
    X_test = seg(df_pre, train, valid)[2]
    result = []
    for C in tqdm(np.logspace(-5,4,10, base=10)):
        lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto', C=C)
        lgfitT = lg.fit(X_train, train['CATEGORY'])
        lgfitV = lg.fit(X_valid, valid['CATEGORY'])
        lgfitTe = lg.fit(X_test, test['CATEGORY'])

        train_pred = score_lg(lgfitT, X_train)
        train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])

        valid_pred = score_lg(lgfitV, X_valid)
        valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])

        test_pred = score_lg(lgfitTe, X_test)
        test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])
        result.append([C, train_accuracy, valid_accuracy, test_accuracy])

    print(f'正解率(学習データ)　:　{train_accuracy:.3f}')
    print(f'正解率(検証データ)　:　{valid_accuracy:.3f}')
    print(f'正解率(評価データ)　:　{test_accuracy:.3f}')
    '''
100%|██████████| 10/10 [03:03<00:00, 39.83s/it]
正解率(学習データ)　:　0.822
正解率(検証データ)　:　0.816
正解率(評価データ)　:　1.000
'''
    result = np.array(result).T
    # print(result)
    plt.plot(result[0], result[1], label='train')
    plt.plot(result[0], result[2], label='valid')
    plt.plot(result[0], result[3], label='test')
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.xlabel('C')
    plt.legend()
    plt.show()

