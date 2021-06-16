import bhtsne
import matplotlib.pyplot as plt
embedded = bhtsne.tsne(np.array(country_vec).astype(np.float64), dimensions=2)
plt.figure(figsize=(15, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, country_name):
    plt.annotate(name, (x, y))
plt.show()