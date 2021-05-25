#!/bin/sh
cat ./dat/popular-names.txt | cut -f1 | sort | uniq -c | sort -rk1 > out_sh/19.txt
