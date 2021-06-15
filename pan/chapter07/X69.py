# 69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．


from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)

    countries = pd.read_table('/users/kcnco/github/100knock2021/pan/chapter07/countries.txt')
    countries = countries['Short name'].values

    # 国名のベクトルを取り出す
    country_vec = []
    country_name = []
    for country in countries:
        if country in model.vocab:
            country_vec.append(model[country])
            country_name.append(country)

    # t-SNEで可視化する
    tsne = TSNE(random_state = 0)
    embs = tsne.fit_transform(country_vec)
    plt.scatter(embs[:, 0], embs[:, 1])
    plt.show()