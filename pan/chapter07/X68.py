# 68. Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
# さらに，クラスタリング結果をデンドログラムとして可視化せよ．

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram

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

    # Ward法による階層型クラスタリング
    linkage_result = linkage(country_vec, method = 'ward', metric = 'euclidean')

    # デンドログラムで結果を表示する
    plt.figure(figsize=(16, 9))
    dendrogram(linkage_result, labels=country_name)
    plt.show()