import bhtsne
from matplotlib import pyplot as plt
import numpy as np
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

embedded = bhtsne.tsne(np.array(countries_vec).astype(np.float64), dimensions=2, rand_seed=123)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, countries):
    plt.annotate(name, (x, y))
plt.show()
