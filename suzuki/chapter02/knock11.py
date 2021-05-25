# -*- coding: utf-8 -*-
#タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

import sys

target = open("./popular-names.txt","r")

for line in target:
    space = line.replace('\t', ' ')
    print(space)

#unixコマンド（sedコマンド）
#sed -e 's/\t/ /g' popular-names.txt

# trコマンド 
#tr '\t' ' ' < popular-names.txt