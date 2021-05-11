import sys
count=0
with open('popular-names.txt') as f:
    for line in f:
        a=line.split(" ")
        for word in line:
            if word=="  ":
                word=" "
        line=" ".join(a)
        