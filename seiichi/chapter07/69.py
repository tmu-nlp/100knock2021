import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import figure
figure(figsize=(20, 10), dpi=80)

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

# https://gist.github.com/cupnoodlegirl/ba10cf7a412a1840714c
countries = pd.read_csv("./data/country_list.csv")
countries = list(countries["ISO 3166-1に於ける英語名"].array)

country_vecs = []
country_names = []
for country in countries:
    if country in vec:
        country_vecs.append(vec[country])
        country_names.append(country)

tsne = TSNE(random_state=0)
embedded = tsne.fit_transform(country_vecs)
plt.scatter(embedded[:, 0], embedded[:, 1])
for (x, y), name in zip(embedded, country_names):
    plt.annotate(name, (x, y))
plt.savefig("data/tsne.png")