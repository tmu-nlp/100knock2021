# 61. 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ．

import numpy as np
from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)
    # コサイン類似度を計算する
    cos_sim = model.similarity('United_States', 'U.S.')

    print(cos_sim)