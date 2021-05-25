#単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

from knock30 import make_morpheme
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

f = open('neko.txt.mecab', 'r')
m = make_morpheme(f)
dic_surface = defaultdict(lambda: 0)

for i in m:
    if i['pos'] != '記号':
        dic_surface[i['surface']] += 1

f.close()

values = dic_surface.values()
frecency = sorted(values, reverse = True)
rank = list(range(1, len(frecency)+1))

#確認
print(len(frecency))
print(len(rank))

plt.rcParams["font.family"] = "Hiragino sans"
plt.scatter(rank, frecency)
plt.title("単語の出現頻度")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("順位")
plt.ylabel("出現頻度")
plt.show()