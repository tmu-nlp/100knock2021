import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors
model=KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True)

'''
element in ws looks like
['word1' , 'word2 , human(mean) , model.similarity]
'''
ws=[]
with open("./wordsim353/combined.csv",'r') as f:
    for line in f:
        line=line.strip()
        if line!="Word 1,Word 2,Human (mean)":
            line=line.split(',')
            line.append(model.similarity(line[0],line[1]))
            ws.append(line)

human=np.array(ws).T[2]
w2v=np.array(ws).T[3]
cor,pv=spearmanr(human,w2v)

print(cor)
'''
0.6849564489532377
'''
