from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models import KeyedVectors


f = open('ans-knock64.txt', 'r')

#国名のセットをアナロジーのファイルから取得
countries = set()
for line in f:
    l = line.strip().split(' ')

    if l[2] in countries or l[1] in countries:
        continue
    elif 'capital-common-countries' in l[0] or 'capital-world' in l[0]:
        countries.add(l[2])
    elif 'currency' in l[0] or  'gram6-nationality-adjective' in l[0]:
        countries.add(l[1])

#国名の単語ベクトルを取得
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
countries = list(countries)
countries_vec = [model[country] for country in countries]

#グラフ化
plt.figure(figsize=(15, 5))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
plt.show()