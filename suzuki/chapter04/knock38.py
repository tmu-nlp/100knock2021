#単語の出現頻度のヒストグラムを描け．
#ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
#縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

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

ans = dic_surface.values()

#確認
#print(ans)

plt.rcParams["font.family"] = "Hiragino sans"
plt.title("単語の出現頻度")
plt.xlabel("出現頻度")
plt.ylabel("単語の異なり数")
plt.hist(ans, bins = 100)
plt.show()