import sys
count=0
a=open('col1.txt','w')
b=open('col2.txt','w')
with open('popular-names.txt') as f:
    for line in f:
        l=line.split(" ")
        for word in line:
            a.write(str(word[0]))
            b.write(str(word[1]))
a.close()
b.close()