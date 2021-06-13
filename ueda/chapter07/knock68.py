import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from knock67 import country_vector

country, country_name = country_vector()
plt.figure(figsize=(20, 10))
dendrogram(linkage(country, method='ward'), labels=country_name)
plt.show()