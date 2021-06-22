""'''
[description]単語の類似度
“United States”と”U.S.”のコサイン類似度を計算せよ．
'''

import gensim
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':

    filepath = './data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

    sim = cos_sim(model['United_States'], model['U.S.'])
    print('sim:{}'.format(sim))

'''
sim:0.7310774922370911
'''