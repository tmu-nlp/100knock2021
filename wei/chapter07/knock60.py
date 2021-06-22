""'''
[description]単語ベクトルの読み込みと表示
学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，”United States”の単語ベクトルを表示せよ．
ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ
'''

import warnings
from gensim.models import KeyedVectors


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
model = KeyedVectors.load_word2vec_format(
    './data/GoogleNews-vectors-negative300.bin',
    binary=True)
us_vec = model['United_States']
print(us_vec)   # 300-dim vector