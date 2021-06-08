import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

with open("./data/questions-words.txt") as f:
    lines = f.readlines()
lines = [line.rstrip("\n").split() for line in lines]

columns = ["category", "v1", "v2", "v3", "v4"]
datas = [[] for _ in range(5)]
for line in lines:
    if line[0] == ":":
        category = line[1]
        continue
    datas[0].append(category)
    for i in range(4):
        datas[i + 1].append(line[i])
df = pd.DataFrame(
    data={"category": datas[0], "v1": datas[1], "v2": datas[2], "v3": datas[3], "v4": datas[4]},
    columns=columns
)

def culculate_similarity(row):
    positive = [row["v2"], row["v3"]]
    negative = [row["v1"]]
    return pd.Series(list(vec.most_similar(positive=positive, negative=negative))[0])

df[["SimWord", "Score"]] = df.progress_apply(culculate_similarity, axis=1)
df.to_csv("./data/64.csv", index=False, header=None)