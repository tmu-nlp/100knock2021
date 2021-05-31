# -*- coding: utf-8 -*-
#自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ．

n = int(input('自然数Nを指定：'))
f = open("./popular-names.txt","r")

for i, line in enumerate(f):
    i += 1
    line = line.strip()
    print(line)
    if i == n:
        break

#unixコマンド
#head -n <数字> <ファイル名>