import gensim
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

def most_similar(tar):
    d = {}
    for k in vec.index2entity:
        d[k] = cos_sim(tar, vec[k])
    sd = reversed(sorted(d.items(), key=lambda x:x[1]))
    return sd

tar = vec["Spain"] - vec["Madrid"] + vec["Athens"]
d = most_similar(tar)
for i, (k, v) in d:
    if i == 10: break
    print(i, k, v)
    
# check
print(model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10))
    
"""
0 Athens 0.7528455
1 Greece 0.6685472
2 Aristeidis_Grigoriadis 0.54957783
3 Ioannis_Drymonakos 0.53614575
4 Greeks 0.5351787
5 Ioannis_Christou 0.5330226
6 Hrysopiyi_Devetzi 0.5088489
7 Iraklion 0.5059265
8 Greek 0.5040616
9 Athens_Greece 0.5034109
"""