import sys

with open('popular-names.txt') as f:
    x = f.readlines()[::-1]
    for i in range(int(sys.argv[1]))[::-1]:
        print(x[i].strip())
