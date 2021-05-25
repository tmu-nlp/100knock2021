# -*- coding: utf-8 -*-
#1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．

f = open("popular-names.txt", "r")
s = set()

for line in f:
    line = line.strip()
    l = line.split("\t")
    s.add(l[0])

print(len(s))

#unixコマンド
#cut -f 1 popular-names.txt | sort | uniq | wc -l