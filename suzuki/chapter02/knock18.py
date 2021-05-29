# -*- coding: utf-8 -*-
#各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．
#確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．

f = open("popular-names.txt", "r")
lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    l = line.split("\t")
    lines[i] = l

lines = sorted(lines, reverse = True, key = lambda x: int(x[2]))

for line in lines:
    print(line)