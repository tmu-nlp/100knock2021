#学習
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    start = time.time()

    # 正解ラベルのみを取り出す
    X_train = pd.read_table('train.feature.txt', header=None)
    Y_train = pd.read_table('train.txt', header = None)[1]                

    # モデルを学習する
    # penalty: 正則化手法
    # solver: 最適解の探索手法
    clf = LogisticRegression(penalty = 'l2', solver = 'sag', random_state = 0)
    clf.fit(X_train, Y_train)

    # モデルを保存する
    joblib.dump(clf, 'model.joblib')

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f'{elapsed_time} [sec]')