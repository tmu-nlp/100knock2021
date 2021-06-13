from scipy.stats import spearmanr
import pandas as pd
from knock60 import struct_vector

model = struct_vector()
wordlist = []
humanlist = []
df = pd.read_csv(r'C:\Git\combined.csv', encoding='utf-8')
for index, row in df.iterrows():
    word_sim = model.similarity(row['Word 1'], row['Word 2'])
    wordlist.append(word_sim)
    humanlist.append(float(row['Human (mean)']))
correlation, pvalue = spearmanr(wordlist, humanlist)
print("Spearmanr: {}".format(correlation))

