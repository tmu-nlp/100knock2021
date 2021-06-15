from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt


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

tsne = TSNE()
tsne.fit(countries_vec)

plt.figure(figsize=(15, 15), dpi=300)
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1])
for (x, y), name in zip(tsne.embedding_, countries):
    plt.annotate(name, (x, y))
plt.show()