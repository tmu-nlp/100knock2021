from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models import KeyedVectors

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

with open('./tmp/knock64.txt') as f:
    lines = f.readlines()

x = set()

for line in lines:
    words = line.split()
    if words[0] in ['capital-common-countries', 'capital-world']:
        x.add(words[2])
    elif words[0] in ['currency', 'gram6-nationality-adjective']:
        x.add(words[1])

countries = list(x)
countries_vec = [MODEL[country] for country in countries]

# ------------------------------------------------------------------------------

plt.figure(figsize=(15, 5))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
plt.show()
