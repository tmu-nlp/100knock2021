from gensim.models import KeyedVectors
import zipfile
from scipy.stats import spearmanr

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

with zipfile.ZipFile('./wordsim353.zip') as zip_f:
    zip_f.extractall('./')

vector = []
human = []
with open('./combined.csv') as f:
    next(f) #１行目はスキップ
    for line in f:
        line = line.split(',')
        sim = model.similarity(line[0], line[1])
        vector.append(sim)
        human.append(line[2])

spe_cor = spearmanr(vector, human)
print(f'スピアマン相関係数: {spe_cor[0]}')

'''
出力結果
スピアマン相関係数: 0.6849564489532376
'''
