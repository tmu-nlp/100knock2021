import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

model=KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True)
country=set()
with open('ans.txt','r') as f:
    for line in f:
        line=line.split()
        if line[0] in  ['capital-common-countries','capital-world']:
            country.add(line[2])
country=list(country)

country_v=[model[c] for c in country]

KM=KMeans(n_clusters=5)
KM.fit(country_v)
for i in range(5):
    cluster=np.where(KM.labels_==i)[0]#return the index of word which belongs to cluster i
    #print('cluster{}'.format(i))
    #print(' '.join([country[e] for e in cluster]))





