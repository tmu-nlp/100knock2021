import zipfile
from gensim.models import KeyedVectors
import numpy as np
from scipy.stats import spearmanr

f = zipfile.ZipFile('wordsim353.zip', 'r')
g = f.open('combined.csv')

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
ws = []

for i, data in enumerate(g):
    if i == 0: 
        continue

    l = data.decode('UTF-8').strip().split(',')
    l.append(model.similarity(l[0], l[1]))
    ws.append(l)

# スピアマン相関係数の計算
human = np.array(ws).T[2] #人間の作成した評価ランキング
w2v = np.array(ws).T[3] #ベクトル計算の評価ランキング
correlation, pvalue = spearmanr(human, w2v)

print(f'スピアマン相関係数: {correlation:.3f}')

'''

スピアマン相関係数: 0.685

'''