#!/bin/sh

cut -f 1 popular-names.txt > col1.txt
cut -f 2 popular-names.txt > col2.txt

###ANS###
#defaultの区切り文字はきっとタブ文字なので、何もしなくてもこれで切れる可能性。
