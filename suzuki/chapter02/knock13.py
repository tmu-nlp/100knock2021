# -*- coding: utf-8 -*-
#12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ．

f1 = open("col1.txt", "r")
f2 = open("col2.txt", "r")
f = open("merge.txt", "w")

for name, sex in zip(f1, f2):
    name = name.strip()
    sex = sex.strip()
    f.write("{}\t{}\n".format(name, sex))

f.close()

#unixコマンド
#paste col1.txt col2.txt > merge_test.txt （区切り文字を指定しなくても勝手にタブ区切りになる）
#diff -s merge_test.txt merge.txt
