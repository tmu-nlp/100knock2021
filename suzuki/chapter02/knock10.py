# -*- coding: utf-8 -*-
#行数をカウントせよ．確認にはwcコマンドを用いよ．

import sys

target = open("./popular-names.txt","r")

n = 0
for line in target:
    n += 1

print(n)
#出力 -> 2780

#unixコマンド
#wc -l popular-names.txt
#出力 -> 2779 popular-names.txt
#wc -l では改行の数を数えるので最後に改行のないファイルでは1行少ない