#予測
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X_test = pd.read_table('test.feature.txt', header = None)
    Y_test = pd.read_table('test.txt', header = None)[1]

    clf = joblib.load('model.joblib')

    # 予測を行う
    Y_predict = clf.predict(X_test)     # 予測したカテゴリのリスト
    Y_proba = clf.predict_proba(X_test) # それぞれのカテゴリである確率のリスト

    # 予測したカテゴリとそれぞれのカテゴリである確率を表示する
    for i, proba in enumerate(Y_proba):
        print(f'predict:{Y_predict[i]}\tb:{proba[0]} e:{proba[1]} m:{proba[2]} t:{proba[3]}')