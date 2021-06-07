#適合率，再現率，F1スコアの計測
#52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ
#カテゴリごとに適合率，再現率，F1スコアを求め
#カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ

import joblib
import pandas as pd
from sklearn.metrics import recall_score,precision_score,f1_score

if __name__ =='__main__':
    X_test = pd.read_table("test.feature.txt", header = None)
    Y_test = pd.read_table("test.txt", header = None)[1]

    # モデルを読み込む
    clf = joblib.load('model.joblib')

    # カテゴリを予測する
    Y_pred = clf.predict(X_test)

    # 適合率、再現率、F1スコアを求める
    rec = recall_score(Y_test, Y_pred, average = None)
    prec = precision_score(Y_test, Y_pred, average = None)
    f1 = f1_score(Y_test, Y_pred, average = None)

    # 適合率、再現率、F1スコアのカテゴリごとのマイクロ平均、マクロ平均を求める
    rec_micro = recall_score(Y_test, Y_pred, average = 'micro')
    rec_macro = recall_score(Y_test, Y_pred, average = 'macro')
    prec_micro = precision_score(Y_test, Y_pred, average = 'micro')
    prec_macro = precision_score(Y_test, Y_pred, average = 'macro')
    f1_micro = f1_score(Y_test, Y_pred, average = 'micro')
    f1_macro = f1_score(Y_test, Y_pred, average = 'macro')

    # 結果を表示する
    print(f'適合率  :{rec}\tマイクロ平均:{rec_micro}\tマクロ平均:{rec_macro}')
    print(f'再現率  :{prec}\tマイクロ平均:{prec_micro}\tマクロ平均:{prec_macro}')
    print(f'F1スコア:{f1}\tマイクロ平均:{f1_micro}\tマクロ平均:{f1_macro}')