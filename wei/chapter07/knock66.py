""'''
[description]WordSimilarity-353での評価
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ.
データは、単語のペアに対して人間が評価した類似度が付与されている。combined.csvは次のフォーマットで
Word 1,Word 2,Human (mean)
love,sex,6.77
tiger,cat,7.35
tiger,tiger,10.00
book,paper,7.46
'''

import gensim
import numpy as np
from scipy.stats import spearmanr



filepath = './data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
ws353 = []
with open('./data/combined.csv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        line = [s.strip() for s in line.split(',')]
        # 1ペアに対して単語ベクトルの類似度を計算して、行の末尾に追記
        line.append(model.similarity(line[0], line[1]))
        ws353.append(line)

for i in range(5):
    print(ws353[i])

# スピアマン相関係数を計算
human = np.array(ws353).T[2]
w2v = np.array(ws353).T[3]
correlation, pvalue = spearmanr(human, w2v)

print(f'スピアマン相関係数:{correlation:.3f}', f'Pvalue:{pvalue:.3f}')

'''
['love', 'sex', '6.77', 0.2639377]
['tiger', 'cat', '7.35', 0.5172962]
['tiger', 'tiger', '10.00', 0.99999994]
['book', 'paper', '7.46', 0.3634626]
['computer', 'keyboard', '7.62', 0.39639163]
スピアマン相関係数:0.685 Pvalue:0.000

'''

