from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from knock67 import KM,country

model=KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True)

tsne = TSNE()
tsne.fit([model[c] for c in country])
cmap = plt.get_cmap('Set1')
plt.figure(figsize=(15, 15), dpi=300)
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1])
for i, ((x, y), name) in enumerate(zip(tsne.embedding_, country)):
    plt.annotate(name, (x, y), color=cmap(KM.labels_[i]))
plt.show()