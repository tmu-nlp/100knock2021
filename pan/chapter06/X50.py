#データの入手・整形
import pandas as pd
import collections
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    newsCorpora_path = '/users/kcnco/github/100knock2021/pan/chapter06/newsCorpora.csv'
    newsCorpora = pd.read_csv(newsCorpora_path, header=None, sep="\t")

    # 列の名前
    colums_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
    newsCorpora.columns = colums_name

    # 抽出
    newsCorpora = newsCorpora[newsCorpora['PUBLISHER'].isin(['Reuters','Huffington Post','Businessweek','Contactmusic.com','Daily Mail'])]

    # 抽出された事例をランダムに並び替える
    # frac: 抽出する行・列の割合を指定
    # random_state: 乱数シードの固定
    newsCorpora = newsCorpora.sample(frac = 1,random_state = 0)

    # X = "TITLE" から Y = "CATEGORY" を予測する
    X = newsCorpora['TITLE']
    Y = newsCorpora['CATEGORY']

    # train:valid:test = 8:1:1 にしたい
    # まず、全体を train:(valid + test) = 8:2 に分ける
    # 次に、(valid + test) を valid:test = 5:5 に分ける
    # stratify: 層化抽出（元のデータの比率と同じになるように分ける）
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, stratify = Y, random_state = 0)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, stratify = Y_test, random_state = 0)

    # X_train と Y_train を列方向に連結する
    # axis: 連結方向
    XY_train = pd.concat([X_train, Y_train], axis = 1)
    XY_valid = pd.concat([X_valid, Y_valid], axis = 1)
    XY_test = pd.concat([X_test, Y_test], axis = 1)

    XY_train.to_csv('/users/kcnco/github/100knock2021/pan/chapter06/train.txt', sep='\t', index = False, header = None)
    XY_valid.to_csv('/users/kcnco/github/100knock2021/pan/chapter06/valid.txt', sep='\t', index = False, header = None)
    XY_test.to_csv('/users/kcnco/github/100knock2021/pan/chapter06/test.txt', sep='\t', index = False, header = None)

    # 学習データ、検証データ、評価データの事例数を確認する
    print(collections.Counter(Y_train))
    print(collections.Counter(Y_valid))
    print(collections.Counter(Y_test))
