import gensim
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

sim = cos_sim(vec["United_States"], vec["U.S."])
print("sim: {}".format(sim))

# sim: 0.7310774922370911