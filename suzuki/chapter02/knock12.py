# -*- coding: utf-8 -*-
#各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．

import sys

target = open("./popular-names.txt","r")
f1 = open("col1.txt", "w")
f2 = open("col2.txt", "w")

for line in target:
    line = line.strip()
    l = line.split("\t")
    f1.write("{}\n".format(l[0]))
    f2.write("{}\n".format(l[1]))

f1.close()
f2.close()

#unixコマンド
#cut -f 1 popular-names.txt > col1_test.txt
#diff -s col1_test.txt col1.txt
#同一なら "Flies <テストファイル> and <作成したファイル> are identical" と表示される