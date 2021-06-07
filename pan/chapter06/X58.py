#正則化パラメータの変更
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    start = time.time()

    X_train = pd.read_table('train.feature.txt', header = None)
    Y_train = pd.read_table('train.txt', header = None)[1]
    X_valid = pd.read_table('valid.feature.txt', header = None)
    Y_valid = pd.read_table('valid.txt', header = None)[1]
    X_test = pd.read_table('test.feature.txt', header = None)
    Y_test = pd.read_table('test.txt', header = None)[1]

    # 正則化パラメータを変えながらモデルを学習する
    C_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    train_acc = []
    valid_acc = []
    test_acc = []
    for c in C_candidate:
        clf = LogisticRegression(penalty = 'l2', solver = 'sag', random_state = 0, C = c)
        clf.fit(X_train, Y_train)
        train_acc.append(accuracy_score(Y_train, clf.predict(X_train)))
        valid_acc.append(accuracy_score(Y_valid, clf.predict(X_valid)))
        test_acc.append(accuracy_score(Y_test, clf.predict(X_test)))

    # 正解率を表示する
    print(train_acc)
    print(valid_acc)
    print(test_acc)

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f'{elapsed_time} [sec]')

    # 正解率をグラフとして表示する
    plt.plot(C_candidate, train_acc, label='train')
    plt.plot(C_candidate, valid_acc, label='valid')
    plt.plot(C_candidate, test_acc, label='test')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()