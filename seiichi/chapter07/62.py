import gensim
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)


tar = vec["United_States"]
d = {}
for k in vec.index2entity:
    d[k] = cos_sim(tar, vec[k])
sd = reversed(sorted(d.items(), key=lambda x:x[1]))
for i, (k, v) in enumerate(sd):
    if i == 10: break
    print(i, k, v)

# check
print(vec.most_similar("United_States", topn=10))

"""
0 United_States 1.0
1 Unites_States 0.7877249
2 Untied_States 0.75413704
3 United_Sates 0.7400725
4 U.S. 0.7310775
5 theUnited_States 0.6404394
6 America 0.617841
7 UnitedStates 0.61673117
8 Europe 0.6132989
9 countries 0.60448045
"""