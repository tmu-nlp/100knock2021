from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from knock67 import country_v,country

plt.figure(figsize=(15, 5))
Z = linkage(country_v, method='ward')
dendrogram(Z, labels=country)
plt.show()