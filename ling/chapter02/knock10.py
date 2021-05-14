import sys
count=0
with open('popular-names.txt') as f:
    for line in f:
        count+=1

print(count)