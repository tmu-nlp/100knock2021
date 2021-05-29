#文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

from knock30 import make_morpheme
from collections import defaultdict

f = open('neko.txt.mecab', 'r')
m = make_morpheme(f)
dic_surface = defaultdict(lambda: 0)

for i in m:
    if i['pos'] != '記号':
        dic_surface[i['surface']] += 1

ans = sorted(dic_surface.items(), key = lambda x:x[1], reverse = True)

#確認
for i in ans[:10]:
    print(i)