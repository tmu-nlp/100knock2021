import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

def culcSimScore(row):
    word1 = row["Word 1"]
    word2 = row["Word 2"]
    if word1 in vec and word2 in vec:
        score = vec.similarity(word1, word2)
    else:
        score = None
    return score


tqdm.pandas()
df = pd.read_csv("./data/wordsim353/combined.csv")
df["SimScore"] = df.progress_apply(culcSimScore, axis=1)

print(df[["Human (mean)", "SimScore"]].corr(method="spearman"))


"""
              Human (mean)  SimScore
Human (mean)      1.000000  0.700017
SimScore          0.700017  1.000000
"""