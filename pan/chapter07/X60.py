# 60. 単語ベクトルの読み込みと表示
# Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，”United States”の単語ベクトルを表示せよ．
# ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ．

from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)
    # 'United_States'の単語ベクトルを取り出す
    us = model['United_States']

    print(us)