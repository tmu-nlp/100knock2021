#出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

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

words = sorted(dic_surface.items(), key = lambda x:x[1], reverse = True)
ans = np.array([i[1] for i in words[:10]])
left = np.array([i[0] for i in words[:10]])

#確認
print(ans)
print(left)

plt.rcParams["font.family"] = "Hiragino sans"
fig = plt.figure()
ax = fig.add_subplot(111, title = '出現頻度の高い単語', ylabel = '頻度', xlabel = '単語')
plt.bar(left, ans)
plt.show()