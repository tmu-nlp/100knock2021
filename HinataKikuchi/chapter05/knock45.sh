#!/bin/sh

# cat ./knock45.txt | sort | uniq -c | sort -nr
cat ./knock45.txt | grep '行う' | sort | uniq -c | sort -nr