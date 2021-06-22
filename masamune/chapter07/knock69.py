import pickle
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

with open('./countries_vec.pickle', mode='rb') as f1, open('./countries.pickle', mode='rb') as f2:
    countries_vec = pickle.load(f1)
    countries = pickle.load(f2)

tsne = TSNE(random_state=0)
vec = tsne.fit_transform(countries_vec)

plt.figure(figsize=(8, 5))
for i in range(len(vec)):
    plt.scatter(vec[i].tolist()[0], vec[i].tolist()[1])
    plt.annotate(list(countries)[i], (vec[i].tolist()[0], vec[i].tolist()[1]))
plt.savefig('./69.png')
plt.show()