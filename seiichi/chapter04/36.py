import pickle
from collections import Counter
import matplotlib.pyplot as plt

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

sf = [d['surface'] for sent in m for d in sent]
cnt = Counter(sf).most_common()

ws, ns = [w for w, _ in cnt[:10]], [n for _, n in cnt[:10]]
plt.bar(ws, ns)
plt.title('頻度上位')
plt.show()
