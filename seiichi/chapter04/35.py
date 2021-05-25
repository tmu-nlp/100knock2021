import pickle
from collections import Counter

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

sf = [d['surface'] for sent in m for d in sent]
cnt = Counter(sf).most_common()

for _, (w, n) in zip(range(10), cnt):
    print(w, n)
