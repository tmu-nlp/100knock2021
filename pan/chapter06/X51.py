#特徴量抽出
import re
import time
import joblib
import numpy as np
import pandas as pd
from stemming.porter2 import stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def preprocessor(text):
    text = text.lower()                          # 文字を全て小文字にする
    text = re.sub(r'\d+', '0', text)             # 数字を全て0にする
    text = ' '.join(word_tokenize(text))         # word_tokenize()を適用するとリストが返ってくるので、' '.join()で結合する
    text = stem(text)

    return text

if __name__ == '__main__':
    start = time.time()                          #開始の時間

    XY_train = pd.read_csv('train.txt', header = None, sep = '\t')
    XY_valid = pd.read_csv('valid.txt', header = None, sep = '\t')
    XY_test = pd.read_csv('test.txt', header = None, sep = '\t')

    columns_name = ['TITLE', 'CATEGORY']
    XY_train.columns = columns_name
    XY_valid.columns = columns_name
    XY_test.columns = columns_name

    # TITLEの文字列に対して、前処理を行う
    XY_train['TITLE'] = XY_train['TITLE'].apply(preprocessor)
    XY_valid['TITLE'] = XY_valid['TITLE'].apply(preprocessor)
    XY_test['TITLE'] = XY_test['TITLE'].apply(preprocessor)

    # TF-IDFを計算する
    vectorizer = TfidfVectorizer(use_idf = True, norm='l2', smooth_idf = True)
    vectorizer.fit(XY_train['TITLE']) 
    train_tfidf = vectorizer.transform(XY_train['TITLE'])
    valid_tfidf = vectorizer.transform(XY_valid['TITLE'])
    test_tfidf = vectorizer.transform(XY_test['TITLE'])

    pd.DataFrame(data=train_tfidf.toarray()).to_csv('train.feature.txt', sep = '\t', index = False, header = None)
    pd.DataFrame(data=valid_tfidf.toarray()).to_csv('valid.feature.txt', sep = '\t', index = False, header = None)
    pd.DataFrame(data=test_tfidf.toarray()).to_csv('test.feature.txt', sep = '\t', index = False, header = None)
    joblib.dump(vectorizer.vocabulary_, 'vocabulary_.joblib') # knock57で使用する


    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f'{elapsed_time} [sec]') 