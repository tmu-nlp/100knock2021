# -*- coding: utf-8 -*-
#各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．

from collections import defaultdict
from os import lseek

f = open("popular-names.txt", "r")
d = defaultdict(lambda: 0)

for line in f:
    line = line.strip()
    l = line.split("\t")
    d[l[0]] += 1

d_sorted = sorted(d.items(), reverse = True, key = lambda x:x[1])

for i in d_sorted:
    print("{}\t{}".format(i[0].ljust(10, " "), i[1]))