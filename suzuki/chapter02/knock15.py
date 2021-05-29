# -*- coding: utf-8 -*-
#自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．

n = int(input('自然数Nを指定：'))
f = open("popular-names.txt", "r")

lines = f.readlines()#dequeを使うと新しくリストを作る必要がない

for line in lines[-n:]:
    print(line.strip())

#unixコマンド
#tail -n <数字> <ファイル名>