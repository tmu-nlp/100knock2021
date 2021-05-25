'''
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
'''
import math
import matplotlib.pyplot as plt
import numpy as np
from knock35 import count
from collections import defaultdict

ranks = [i for i in range(len(count))]
values = [w[1] for w in count]
plt.figure(figsize=(8, 4))
plt.scatter(ranks, values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.show()