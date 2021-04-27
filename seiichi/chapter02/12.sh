#!/bin/sh
cat ./dat/popular-names.txt | cut -f 1 > out_sh/col1.txt
cat ./dat/popular-names.txt | cut -f 2 > out_sh/col2.txt
