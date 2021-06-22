""'''
[description]t-SNEによる可視化
国名に関する単語ベクトルのベクトル空間をt-SNEで可視化せよ'''

import gensim
import bhtsne
import numpy as np
from matplotlib import pyplot as plt


filepath = './data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

# 単語アナロジーの評価データから、適当な国名リストを取得
countries = set()
with open('./knock64_add_sim.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        if line[0] in ['capital-common-countries', 'capital-world']:
            countries.add(line[2])
        elif line[0] in ['currency', 'gram6-nationality-adjective']:
            countries.add(line[1])
countries = list(countries)

# 単語ベクトルを取得
countries_vec = [model[country] for country in countries]

embedded = bhtsne.tsne(np.array(countries_vec).astype(np.float64), dimensions=2, rand_seed=123)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, countries):
    plt.annotate(name, (x, y))
plt.show()



