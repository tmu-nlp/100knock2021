import pickle
from collections import Counter
import matplotlib.pyplot as plt

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

co = []
for sent in m:
    if any(d['base'] == '猫' for d in sent):
        w = [d['base'] for d in sent if d['base'] != '猫']
        co += w
cnt = Counter(co).most_common()

ws, ns = [w for w, _ in cnt[:10]], [n for _, n in cnt[:10]]
plt.bar(ws, ns)
plt.title('頻度上位')
plt.show()
