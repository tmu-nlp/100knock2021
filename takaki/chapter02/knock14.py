import sys

with open('popular-names.txt') as f:
    for i in range(int(sys.argv[1])):
        print(f.readline().strip())
