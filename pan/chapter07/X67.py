# 67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors

# クラスタ数k=5として
CLUSTER_NUM = 5
# read_tableを使ってデータを読み込んで、short name列を取得して
countries = pd.read_table('/users/kcnco/github/100knock2021/pan/chapter07/countries.txt')
countries = countries['Short name']

model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)

country_vecs = []
country_names = []
for country in countries:
    if country in model:
        # 国名一つずつ抽出し、読み込んだモデルに対してモデル内に国名があれば単語ベクトルと国名をリストに追加して
        country_vecs.append(model[country])
        country_names.append(country)

# k-meansを使って
kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state = 0)
# fit()を用いてデータをクラスタリングして結果を出力して
kmeans.fit(country_vecs)
for i in range(CLUSTER_NUM):
    # where()を用いて各ラベルのデータを出力します
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster:', i)
    # 最終的にクラスタのラベルとjoin()で国名を結合して
    print(', '.join([country_names[j] for j in cluster]))
