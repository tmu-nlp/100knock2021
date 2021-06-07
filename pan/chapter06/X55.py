#混同行列の作成
import time
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    start = time.time()

    X_train = pd.read_table('train.feature.txt', header = None)
    Y_train = pd.read_table('train.txt', header = None)[1]
    X_test = pd.read_table('test.feature.txt', header = None)
    Y_test = pd.read_table('test.txt', header = None)[1]

    clf = joblib.load('model.joblib')

    # 混同行列を求める
    train_cm = confusion_matrix(Y_train, clf.predict(X_train))
    test_cm = confusion_matrix(Y_test, clf.predict(X_test))

    # 混同行列をデータフレームワークに変換する
    # また、カテゴリのラベルを追加する
    labels = ['b' ,'e', 'm', 't']
    train_cm_labeled = pd.DataFrame(train_cm, columns=labels, index=labels)
    test_cm_labeled = pd.DataFrame(test_cm, columns=labels, index=labels)

    # 混同行列を表示する
    print(f'train confusion matrix:')
    print(f'{train_cm_labeled}')
    print(f'test confusion matrix:')
    print(f'{test_cm_labeled}')

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f'{elapsed_time} [sec]') 