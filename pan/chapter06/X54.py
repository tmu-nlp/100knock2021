#正解率の計測
import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    start = time.time()

    X_train = pd.read_table('train.feature.txt', header = None)
    Y_train = pd.read_table('train.txt', header = None)[1]
    X_test = pd.read_table('test.feature.txt', header = None)
    Y_test = pd.read_table('test.txt', header = None)[1]

    clf = joblib.load('model.joblib')

    # 正解率を求める
    train_acc = accuracy_score(Y_train, clf.predict(X_train))
    test_acc = accuracy_score(Y_test, clf.predict(X_test))

    # 正解率を表示する
    print(f'train accuracy: {train_acc}') 
    print(f'test  accuracy: {test_acc}')

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f'{elapsed_time} [sec]')
