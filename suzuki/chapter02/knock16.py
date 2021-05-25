# -*- coding: utf-8 -*-
#自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．同様の処理をsplitコマンドで実現せよ．

import math

n = int(input('自然数Nを指定：'))
f = open("popular-names.txt", "r")

lines = f.readlines()

lines_per_block = math.ceil(len(lines) / n)

for i, line in enumerate(lines):
    print(line.strip())
    if i % lines_per_block == 0:
        print("\n")

#unixコマンド
#split -l <行数> popular-names.txt knock16 （このままだと行数で分けられるのでN分割ではない）
