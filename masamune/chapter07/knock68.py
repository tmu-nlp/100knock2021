import pickle
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

with open('./countries_vec.pickle', mode='rb') as f1, open('./countries.pickle', mode='rb') as f2:
    countries_vec = pickle.load(f1)
    countries = pickle.load(f2)

Z = linkage(countries_vec, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=list(countries))
plt.show()