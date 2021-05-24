#「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

from knock30 import make_morpheme
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

f = open('neko.txt.mecab', 'r')
m = make_morpheme(f)
dic_surface = defaultdict(lambda: 0)

n = 2

for i in range(len(m)): #前後 n 文字で共起語をカウント
    if m[i]['surface'] == '猫':
        for j in range(0, 1 + n*2):
            s = i - n
            if s + j != i and m[s + j]['pos'] != '記号':
                dic_surface[m[s + j]['surface']] += 1
        
f.close()

words = sorted(dic_surface.items(), key = lambda x:x[1], reverse = True)
ans = np.array([i[1] for i in words[:10]])
left = np.array([i[0] for i in words[:10]])

#確認
print(ans)
print(left)

plt.rcParams["font.family"] = "Hiragino sans"
fig = plt.figure()
ax = fig.add_subplot(111, title = '頻度の高い共起語', ylabel = '頻度', xlabel = '単語')
plt.bar(left, ans)
plt.show()