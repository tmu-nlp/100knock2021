import pickle
from collections import Counter
import matplotlib.pyplot as plt

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

bs = [d['base'] for sent in m for d in sent]
cnt = Counter(bs).most_common()
fs = [f for _, f in cnt]

plt.hist(fs, bins=100, range=(1, 100))
plt.show()
