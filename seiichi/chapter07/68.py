import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

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

X = linkage(country_vecs, method="ward", metric="euclidean")
plt.figure(num=None, figsize=(15, 5), dpi=200, facecolor="w", edgecolor="k")
dendrogram(X, labels=country_names)
plt.savefig("./data/dendrogram.png")