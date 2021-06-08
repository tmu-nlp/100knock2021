'''
[task description]特徴量の抽出
特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ
記事の見出しを単語列に変換したものが最低限のベースラインとなる
'''""


import string
import re
import pandas as pd
from knock50 import load_df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)           # 記号をスペースに置換
    text = text.lower()                    # 小文字化
    text = re.sub('[0-9]+', '0', text)     # 数字列を0に置換

    return text

def df_pre(train, valid, test):
    # データの再結合
    df = pd.concat([train, valid, test], axis=0)
    df.reset_index(drop=True, inplace=True)

# 前処理の実施
    df['TITLE'] = df['TITLE'].map(lambda x: preprocessing(x))
    return df


def train_seg(df_pre, train, valid):

    # データの分割
    train_valid = df_pre[:len(train) + len(valid)]
    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,2))
    # ベクトル化
    X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE'])
    # ベクトルをデータフレームに変換
    X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names())
    # データの分割
    X_train = X_train_valid[:len(train)]

    return X_train


def valid_seg(df_pre, train, valid):
    # データの分割
    train_valid = df_pre[:len(train) + len(valid)]
    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
    # ベクトル化
    X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE'])
    # ベクトルをデータフレームに変換
    X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names())
    # データの分割
    X_valid = X_train_valid[len(train):]

    return X_valid


def test_seg(df_pre, train, valid):
    # データの分割
    train_valid = df_pre[:len(train) + len(valid)]
    test = df_pre[len(train) + len(valid):]
    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
    # ベクトル化
    vec_tfidf.fit_transform(train_valid['TITLE'])
    X_test = vec_tfidf.transform(test['TITLE'])

    # ベクトルをデータフレームに変換
    X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names())

    return X_test


if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = train_seg(df_pre, train, valid)
    X_valid = valid_seg(df_pre, train, valid)
    X_test = test_seg(df_pre, train, valid)
    # データの保存
    X_train.to_csv('./X_train.txt', sep='\t', index=False)
    X_valid.to_csv('./X_valid.txt', sep='\t', index=False)
    X_test.to_csv('./X_test.txt', sep='\t', index=False)

    print(X_test.head())
