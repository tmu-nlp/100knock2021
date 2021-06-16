import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(25, 15))
Y = linkage(country_vec, method='ward')
dendrogram(Y, labels=country_name)
plt.show()