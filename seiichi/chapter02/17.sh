#!/bin/sh
cut -f 1 ./dat/popular-names.txt | sort | uniq > out_sh/17.txt
