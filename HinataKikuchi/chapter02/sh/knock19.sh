#!/bin/sh

cut -f1 popular-names.txt | sort | uniq -c | sort -r