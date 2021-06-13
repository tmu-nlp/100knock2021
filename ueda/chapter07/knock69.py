from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from knock67 import country_vector

country, country_name = country_vector()
tsne = TSNE(n_components=2, random_state=2021, perplexity=30, n_iter=1000)
embedded = tsne.fit_transform(country)
kmeans = KMeans(n_clusters=5, random_state=2021).fit_predict(country)
plt.figure(figsize=(10, 10))
colors =  ["r", "g", "b", "c", "m"]
for i in range(embedded.shape[0]):
    plt.scatter(embedded[i][0], embedded[i][1], marker='.', color=colors[kmeans[i]])
    plt.annotate(country_name[i], xy=(embedded[i][0], embedded[i][1]), color=colors[kmeans[i]])
plt.show()